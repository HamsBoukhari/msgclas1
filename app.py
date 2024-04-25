import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel, BertTokenizer, BertForSequenceClassification
import torch
import re
def clean_sequence(input_text):
    pattern = r'[<\"/]'
    cleaned_seq1 = re.sub(pattern, '', input_text)
    cleaned_seq = cleaned_seq1.replace('>',' ')
    return cleaned_seq
def predict(seq,model):
    model.eval()
    input_ids = tokenizer.encode(seq, return_tensors='pt')
    output = model.generate(input_ids, max_length=1000, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    output_sequence = decoded_output[len(seq):].strip()
    return output_sequence
def predict1(text):
    model4.eval()
    encoding = tokenizer1(text, return_tensors='pt', max_length=512, padding='max_length', truncation=True)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    with torch.no_grad():
        outputs = model4(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        _, preds = torch.max(logits, dim=1)
    if preds.item() == 1:
        return "Mark As Give Up"
    else:
        return "Full Service"

@st.cache(allow_output_mutation=True)
def get_model():
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model1 = GPT2LMHeadModel.from_pretrained("hams2/fullservice")
    model2 = GPT2LMHeadModel.from_pretrained("hams2/markasgup1")
    model3 = GPT2LMHeadModel.from_pretrained("hams2/markasgup2")
    tokenizer1 = BertTokenizer.from_pretrained('bert-base-uncased')
    model4 = BertForSequenceClassification.from_pretrained("hams2/class1")
    return tokenizer,model1,model2,model3,tokenizer1,model4

tokenizer,model1,model2,model3,tokenizer1,model4 = get_model()
trade_msg = st.text_area('Trade Message')
sgw_op = st.text_area('SGW Operation')
button = st.button("Predict")
if (trade_msg and sgw_op) and button:
    trade_msg1 = trade_msg.replace('\n','')
    sgw_op1 = sgw_op.replace('\n','')
    input_seq = trade_msg1+' '+sgw_op1
    seq = clean_sequence(input_seq)
    sgw_op2 = clean_sequence(sgw_op1)
    workflow = predict1(sgw_op2)
    if workflow=="Full Service":
        st.write("Workflow: ",workflow)
        ccp_mssg = predict(seq,model1)
        st.write("CCP Message: ",ccp_mssg)
    else:
        st.write("Workflow: ",workflow)
        ccp_mssg1 = predict(seq,model2)
        st.write("CCP Message1: ",ccp_mssg1)        
        ccp_mssg2 = predict(seq,model3)
        st.write("CCP Message2: ",ccp_mssg2) 
