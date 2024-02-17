## TF-Lite Tokenizer
### 10x faster than traditional tokenizers deployable on web and mobile platforms.
Optimized Tokenizer using Tensorflow to auto group queries and tokenize them for faster dynamic batching.

```python
from tensorflow.lite.python import interpreter
import tensorflow_text as text
# Perform TensorFlow Lite inference.
interp = interpreter.InterpreterWithCustomOps(model_content=tflite_model,custom_op_registerers=text.tflite_registrar.SELECT_TFTEXT_OPS)
interp.get_signature_list()
tokenize = interp.get_signature_runner('serving_default')
output = tokenize(input=np.array(["examples"]))
```