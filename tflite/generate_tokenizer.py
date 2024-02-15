# type: ignore
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
import tensorflow as tf
import keras
import tensorflow_text as text
with open('jina/vocab.txt','r') as fp:
    _VOCAB = fp.readlines()
    _VOCAB = [v.strip().encode('utf-8') for v in _VOCAB]

_START_TOKEN = _VOCAB.index(b"[CLS]")
_END_TOKEN = _VOCAB.index(b"[SEP]")
_MASK_TOKEN = _VOCAB.index(b"[MASK]")
_UNK_TOKEN = _VOCAB.index(b"[UNK]")
_MAX_SEQ_LEN = 8192
_VOCAB_SIZE = len(_VOCAB)

@tf.function
def get_best_bucket(b):
    buckets = tf.constant([4,8,16,32,64,128,256,512,1024,2048],tf.int32)
    max_row = tf.cast((tf.math.reduce_max(b.row_lengths())),tf.int32)
    seqc = buckets - max_row
    bucket_idx = tf.argmin(tf.map_fn(blow_up,seqc))
    best_bucket = buckets[bucket_idx]
    return best_bucket

@tf.function
def blow_up(seq):
    if seq<0:
        return tf.cast(2**16,tf.int32)
    else:
        return seq

class TokenLayer(keras.layers.Layer):
    def __init__(self):
        super(TokenLayer,self).__init__()
        self._VOCAB = _VOCAB
        self._START_TOKEN = _VOCAB.index(b"[CLS]")
        self._END_TOKEN = _VOCAB.index(b"[SEP]")
        self._MASK_TOKEN = _VOCAB.index(b"[MASK]")
        self._UNK_TOKEN = _VOCAB.index(b"[UNK]")
        self._MAX_SEQ_LEN = 8192
        self._VOCAB_SIZE = len(_VOCAB)
        self.normalizer = text.FastBertNormalizer(lower_case_nfd_strip_accents=True)
        self.normalizer = self.normalizer._model

    def build(self, input_shape):
        self.tokenizer = text.FastBertTokenizer(
            self._VOCAB,
            # fast_bert_normalizer_model_buffer=self.normalizer,
            lower_case_nfd_strip_accents=True
        )
        
    def call(self,inputs):
        inputs = self.tokenizer.tokenize(inputs)
        # inputs = inputs.merge_dims(-2,-1)
        input_ids,token_type_ids = text.combine_segments(
            [inputs],
            start_of_sequence_id=self._START_TOKEN,
            end_of_segment_id=self._END_TOKEN
        )
        curr_bucket = get_best_bucket(input_ids)
        input_ids, attention_mask = text.pad_model_inputs(input_ids, max_seq_length=curr_bucket)
        token_type_ids,_ = text.pad_model_inputs(token_type_ids, max_seq_length=curr_bucket)
        return input_ids,token_type_ids, attention_mask

tk=TokenLayer()
text_input = keras.layers.Input(shape=(),dtype=tf.string,name="input")
input_ids,token_type_ids, attention_mask = tk(text_input)
outputs = input_ids,token_type_ids, attention_mask
output_names = ['input_ids','token_type_ids', 'attention_mask']
outputs = dict(zip(output_names,outputs))
model = keras.Model(inputs=text_input,outputs= outputs)
model.output_names = output_names
model.compile(jit_compile=True)

a = model(tf.constant(["hello world"]))
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
converter.allow_custom_ops = True
tflite_model = converter.convert()
with open('tokenizer.tflite','wb') as fp:
    fp.write(tflite_model)
print(tf.lite.experimental.Analyzer.analyze(model_content=tflite_model))
from tensorflow.lite.python import interpreter
# Perform TensorFlow Lite inference.
interp = interpreter.InterpreterWithCustomOps(model_content=tflite_model,custom_op_registerers=text.tflite_registrar.SELECT_TFTEXT_OPS)
interp.get_signature_list()
tokenize = interp.get_signature_runner('serving_default')
output = tokenize(input=np.array(["examples"]))
# print('TensorFlow Lite result = ', output)
