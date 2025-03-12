from .evaluator import evaluate, simple_evaluate

### COUPLED ADAM START
from transformers import AutoConfig, AutoModelForCausalLM
from .models.nanogpt.model_hf import MyConfig, MyModel
AutoConfig.register("nanogpt", MyConfig)
AutoModelForCausalLM.register(MyConfig, MyModel)
### COUPLED ADAM END
