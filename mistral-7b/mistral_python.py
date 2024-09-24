from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

tokenizer = MistralTokenizer.from_file("/app/mistral/model/Mistral-7B-Instruct-v0.3/tokenizer.model.v3")  # change to extracted tokenizer file
model = Transformer.from_folder("/app/mistral/model/7B-Instruct/Mistral-7B-Instruct-v0.3")  # change to extracted model dir
model.load_lora("/app/mistral/model/checkpoint-0100/lora.safetensors")

completion_request = ChatCompletionRequest(messages=[UserMessage(content="Item No. 103 And In the matter of: Debasish Banerjee & Ors.- versus -The Petitioners, apprehending arrest in connection with Raiganj Police Station Case No.994 of 2013 dated 12.09.2013 under sections 498A/34 of the Indian Penal Code, 1860, have applied for anticipatory bail.We have heard the learned Advocate for the Petitioners and the learned Advocate for the State.We have seen the case diary. In our opinion, the Petitioner No. 1, Debasish Banerjee, who is the husband of the Complainant, does not deserve to be granted anticipatory bail.As regards the other Petitioners, who are the relatives of the Petitioner No.1, there is no need for their custodial interrogation in this case.Hence, we allow their application and direct that in the event of arrest, the Petitioner Nos. 2 to 5, Partha Sarathi Banerjee, Smt. Minati Banerjee, Smt. Mekhla Banerjee and Smt. Moushumi Banerjee, shall be released on bail upon furnishing a bond `5,000/- (Rupees Five thousand) each with one surety each of like amount to the satisfaction of the Court concerned subject to the conditions laid down under section 438 (2) of the Code of Criminal Procedure, 1973 .The application for anticipatory bail is, thus, disposed of. (Nishita Mhatre, J) (Ranjit Kumar Bag, J)")])

tokens = tokenizer.encode_chat_completion(completion_request).tokens

out_tokens, _ = generate([tokens], model, max_tokens=64, temperature=0.0, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])

print(result)
