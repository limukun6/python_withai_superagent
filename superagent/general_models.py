"""Small modality-driven portfolios with preprocessing inside fitted pipelines."""
from functools import partial
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.svm import SVC

from .schema import MixedRows,TextTableRows,MISSING


class TextValues(TransformerMixin,BaseEstimator):
    def fit(self,x,y=None):return self
    def transform(self,x):
        return [str(v) if str(v).strip() else '__EMPTY_TEXT__' for v in x]


class TextTablePreprocessor(TransformerMixin,BaseEstimator):
    def fit(self,x,y=None):
        if not isinstance(x,TextTableRows):raise ValueError('Text-table data requires its frozen schema')
        self.columns_=x.columns;self.numeric_columns_=x.numeric_columns;self.text_columns_=x.text_columns
        transformers=[]
        if self.numeric_columns_:
            transformers.append(('numeric',Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),
                ('scale',StandardScaler())]),list(self.numeric_columns_)))
        categories=[i for i in range(x.shape[1]) if i not in self.numeric_columns_+self.text_columns_]
        if categories:
            transformers.append(('categories',Pipeline([('impute',SimpleImputer(strategy='most_frequent',keep_empty_features=True)),
                ('encode',OneHotEncoder(handle_unknown='ignore'))]),categories))
        for index in self.text_columns_:
            transformers.append((f'text{index}',Pipeline([('clean',TextValues()),
                ('tfidf',TfidfVectorizer(analyzer='char_wb',ngram_range=(1,4),min_df=1,max_features=20000,sublinear_tf=True))]),index))
        self.transformer_=ColumnTransformer(transformers,sparse_threshold=1.)
        self.transformer_.fit(self._values(x),y)
        return self

    def _values(self,x):
        if not isinstance(x,TextTableRows) or (x.columns,x.numeric_columns,x.text_columns)!=(self.columns_,self.numeric_columns_,self.text_columns_):
            raise ValueError('Text/table columns differ from frozen preprocessing')
        values=x.values.copy()
        for index in range(values.shape[1]):
            if index in self.numeric_columns_:values[:,index]=np.asarray(values[:,index],dtype=float)
            elif index not in self.text_columns_:
                values[:,index]=[np.nan if str(v).strip().casefold() in MISSING else str(v) for v in values[:,index]]
        return values

    def transform(self,x):return self.transformer_.transform(self._values(x))


class MixedPreprocessor(TransformerMixin,BaseEstimator):
    def _values(self,x):
        if not isinstance(x,MixedRows) or x.columns!=self.columns_ or x.numeric_columns!=self.numeric_columns_:
            raise ValueError('Mixed table schema differs from fitted preprocessing')
        values=x.values.copy()
        for i in range(values.shape[1]):
            if i in self.numeric_columns_:values[:,i]=np.asarray(values[:,i],dtype=float)
            else:values[:,i]=[np.nan if str(v).strip().casefold() in MISSING else str(v) for v in values[:,i]]
        return values

    def fit(self,x,y=None):
        if not isinstance(x,MixedRows):raise ValueError('Mixed preprocessing requires a declared schema')
        self.columns_=x.columns;self.numeric_columns_=x.numeric_columns
        other=[i for i in range(x.shape[1]) if i not in self.numeric_columns_]
        self.transformer_=ColumnTransformer([
            ('numeric',Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),
                                 ('scaler',StandardScaler())]),list(self.numeric_columns_)),
            ('categorical',Pipeline([('imputer',SimpleImputer(strategy='most_frequent',keep_empty_features=True)),
                 ('encoder',OneHotEncoder(handle_unknown='ignore'))]),other)],sparse_threshold=1.)
        self.transformer_.fit(self._values(x),y)
        return self

    def transform(self,x):return self.transformer_.transform(self._values(x))


class ImageFlattener(TransformerMixin,BaseEstimator):
    def fit(self,x,y=None):
        x=np.asarray(x)
        if x.ndim not in (3,4):raise ValueError('Expected image tensor N,H,W[,C]')
        self.feature_shape_=x.shape[1:]
        return self

    def transform(self,x):
        x=np.asarray(x)
        if x.shape[1:]!=self.feature_shape_:raise ValueError('Image dimensions differ from training')
        return x.reshape(len(x),-1).astype(float)


def _fit(kind,algorithm,x_train,y_train,x_search,c=1.,leaf=1):
    if kind=='text_tabular':steps=[('features',TextTablePreprocessor())]
    elif kind=='mixed_tabular':steps=[('features',MixedPreprocessor())]
    elif kind=='image':steps=[('flatten',ImageFlattener()),('scale',StandardScaler())]
    else:steps=[('imputer',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),('scale',StandardScaler())]
    if algorithm=='fourier_logistic':
        from .numeric_kernel import NumericKernelFeatures
        steps.append(('kernel', NumericKernelFeatures()))
    if algorithm in ('logistic','fourier_logistic'):classifier=LogisticRegression(C=c,max_iter=2000,random_state=42)
    elif algorithm=='extra_trees':classifier=ExtraTreesClassifier(n_estimators=300,min_samples_leaf=leaf,n_jobs=-1,random_state=42)
    elif algorithm=='rbf_svm':classifier=SVC(C=c,kernel='rbf',gamma='scale')
    else:raise ValueError('Unknown generic algorithm')
    model=Pipeline([*steps,('classifier',classifier)])
    model.fit(x_train,y_train)
    return model,model.predict(x_search)


def _text(x_train,y_train,x_search,representation='word',c=1.):
    word=TfidfVectorizer(ngram_range=(1,2),min_df=1,sublinear_tf=True,max_features=40_000)
    char=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=1,sublinear_tf=True,max_features=40_000)
    features=word if representation=='word' else char if representation=='char' else FeatureUnion([('word',word),('char',char)])
    model=Pipeline([('features',features),('classifier',LogisticRegression(C=c,max_iter=2000,random_state=42))])
    model.fit(x_train,y_train)
    return model,model.predict(x_search)


def generic_candidates(kind):
    from .models import Candidate
    if kind=='single_text':
        return [Candidate(f'single_{representation}_C{c}',{'representation':representation,'C':c},partial(_text,representation=representation,c=c))
                for representation in ('word','char','word_char') for c in (.3,3.)]
    if kind not in ('numeric_tabular','mixed_tabular','text_tabular','image'):raise ValueError('Unknown generic task kind')
    candidates=[Candidate(f'{kind}_logistic_C{c}',{'C':c},partial(_fit,kind,'logistic',c=c)) for c in (.1,1.,10.)]
    candidates += [Candidate(f'{kind}_extra_trees_leaf{leaf}',{'min_samples_leaf':leaf,'n_estimators':300},partial(_fit,kind,'extra_trees',leaf=leaf)) for leaf in (1,3)]
    if kind not in ('mixed_tabular','text_tabular'):candidates += [Candidate(f'{kind}_rbf_svm_C{c}',{'C':c,'kernel':'rbf'},partial(_fit,kind,'rbf_svm',c=c)) for c in (1.,10.)]
    return candidates
