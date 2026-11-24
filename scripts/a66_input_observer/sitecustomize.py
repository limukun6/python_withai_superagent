"""Observe the actual classifier inputs without changing training behavior."""
import os
import sys
if os.environ.get('A66_OBSERVE')=='1' and 'superagent.experiment_worker' in sys.orig_argv:
    import json
    from pathlib import Path
    import numpy as np
    from scipy.sparse import csr_matrix,save_npz
    from sklearn.linear_model import LogisticRegression
    def no_network(event,args):
        if event in ('socket.connect','socket.getaddrinfo','socket.sendto'):raise PermissionError('A66 network forbidden')
    sys.addaudithook(no_network)
    original=LogisticRegression.fit
    def observe(model,x,y,*args,**kwargs):
        folder=Path(os.environ['A66_INPUT_DIR'])
        marker=folder/'classifier_input.json'
        if marker.exists():raise RuntimeError('Unexpected extra supervised fit in A66 worker')
        save_npz(folder/'classifier_input.npz',csr_matrix(x));np.save(folder/'classifier_labels.npy',np.asarray(y),allow_pickle=False)
        marker.write_text(json.dumps(dict(classifier=type(model).__name__,rows=x.shape[0],columns=x.shape[1],status='before_actual_fit')))
        return original(model,x,y,*args,**kwargs)
    LogisticRegression.fit=observe
