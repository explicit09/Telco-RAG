"""Optional local ONNX cross-encoder; scores only supplied retrieval candidates."""
import hashlib
import importlib.metadata
import math
from pathlib import Path

from .models import Evidence


class ONNXReranker:
    def __init__(self, directory, *, batch_size=8, max_length=512):
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer
        if batch_size < 1 or not 8 <= max_length <= 512:
            raise ValueError('invalid reranker batching or token limit')
        folder=Path(directory)
        model, tokenizer = folder/'model.onnx', folder/'tokenizer.json'
        self.provenance = {
            'model_sha256': hashlib.sha256(model.read_bytes()).hexdigest(),
            'tokenizer_sha256': hashlib.sha256(tokenizer.read_bytes()).hexdigest(),
            'runtime_versions': {name:importlib.metadata.version(name) for name in ('onnxruntime','tokenizers','numpy')},
            'max_length':max_length,'batch_size':batch_size,
        }
        self.np=np
        self.batch_size=batch_size
        self.tokenizer=Tokenizer.from_file(str(tokenizer))
        self.tokenizer.enable_truncation(max_length=max_length)
        self.tokenizer.enable_padding(pad_id=0,pad_token='[PAD]')
        options=ort.SessionOptions()
        options.intra_op_num_threads=2
        options.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])

    def rank(self, query, evidence):
        ranked=[]
        for offset in range(0,len(evidence),self.batch_size):
            batch=evidence[offset:offset+self.batch_size]
            encodings=self.tokenizer.encode_batch([(query,e.chunk.text) for e in batch])
            columns={'input_ids':'ids','attention_mask':'attention_mask','token_type_ids':'type_ids'}
            inputs={field.name:self.np.array([getattr(e,columns[field.name]) for e in encodings],dtype=self.np.int64) for field in self.session.get_inputs()}
            scores=self.session.run(None,inputs)[0].reshape(-1)
            if len(scores)!=len(batch) or any(not math.isfinite(float(s)) for s in scores):
                raise ValueError('reranker returned invalid scores')
            ranked.extend(Evidence(e.chunk,float(score),'cross_encoder') for e,score in zip(batch,scores))
        return sorted(ranked,key=lambda e:(-e.score,e.chunk.id))
