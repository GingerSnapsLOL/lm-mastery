from transformers import AutoConfig, AutoModelForCausalLM
from .configuration_alko import AlkoConfig
from .modeling_alko import AlkoLLM

AutoConfig.register("alko", AlkoConfig)
AutoModelForCausalLM.register(AlkoConfig, AlkoLLM)