"""Column-specific features from smoothed category distributions fitted on training rows."""
import numpy as np

from .models import _clean_categories
from .tabular_encoder import LevelClassifier


def column_codes(rows):
    result={}
    for index,column in enumerate(np.char.strip(_clean_categories(rows)).T):
        labels,inverse,counts=np.unique(column,return_inverse=True,return_counts=True)
        if 2<=len(labels)<=128 and 2*len(labels)<=len(column):
            result[index]=(labels,inverse,counts)
    return result


class DistributionClassifier(LevelClassifier):
    def fit(self,rows,labels):
        codes=column_codes(rows)
        n=len(rows)
        proposals=[]
        for index,(levels,inverse,counts) in codes.items():
            if len(levels)<3:continue
            residuals=[]
            for other,(options,context,context_counts) in codes.items():
                if index==other:continue
                table=np.bincount(inverse*len(options)+context,minlength=len(levels)*len(options)).reshape(len(levels),-1)
                marginal=context_counts/n
                conditional=(table+self.params['prior_count']*marginal)/(counts[:,None]+self.params['prior_count'])
                residuals.append(np.sqrt(conditional)-np.sqrt(marginal))
            if not residuals:continue
            matrix=np.column_stack(residuals)
            weights=counts/n
            matrix-=weights@matrix
            weighted=np.sqrt(weights[:,None])*matrix
            dependence=float(np.square(weighted).sum()/len(residuals))
            proposals.append((dependence,index,levels,counts,matrix))
            proposals.sort(key=lambda v:(-v[0],v[1]))
            del proposals[8:]
        self.lookups={}
        selected=[]
        for strength,index,levels,counts,matrix in proposals:
            if strength<=1e-12:continue
            if self.params['representation']=='frequency':
                coords=np.log(counts[:,None]/n)
                coords-=(counts/n)@coords
            else:
                weighted=np.sqrt(counts[:,None]/n)*matrix
                _,_,vt=np.linalg.svd(weighted,full_matrices=False)
                dims=min(3,len(counts)-1,len(vt))
                coords=matrix@vt[:dims].T
                if self.params['representation']=='permuted':
                    cycle=np.random.default_rng(47923+index).permutation(len(coords))
                    altered=coords.copy()
                    altered[cycle]=coords[np.roll(cycle,1)]
                    coords=altered
            self.lookups[index]=dict(zip(levels,coords.astype(np.float32),strict=True))
            selected.append(dict(column=index,levels=len(levels),dimensions=coords.shape[1],dependence=strength))
        self.training_metadata=dict(representation=self.params['representation'],training_rows=n,columns=selected)
        return self.fit_backbone(rows,labels)

    def features(self,rows):
        return super().features(np.char.strip(_clean_categories(rows)))
