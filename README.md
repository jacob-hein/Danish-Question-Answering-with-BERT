# Danish Question Answering with BERT
A pre-trained Danish BERT model was fine-tuned on a machine translated SQuAD-da dataset for the NLP task of Question Answering. The fine-tuned model is available [here on Huggingface.](https://huggingface.co/jacobshein/danish-bert-botxo-qa-squad)

## Load the model
```python
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
tokenizer = AutoTokenizer.from_pretrained("jacobshein/danish-bert-botxo-qa-squad")
model = AutoModelForQuestionAnswering.from_pretrained("jacobshein/danish-bert-botxo-qa-squad")
```

## Application with example context
```python
context = """
Det danske eliteindeks står malet i mørkerødt onsdag eftermiddag, hvor C25-indekset falder over 2,5 pct. 
Lige nu falder 11 af de danske eliteaktier mere end 3 pct. i dagens handel, og aktieindekset rammer det laveste niveau i år. 
Faldene skyldes først og fremmest rotation i markedet, påpeger Jacob Pedersen, aktieanalysechef i Sydbank. 
“Der er noget rotation i gang, og pengene bliver dirigeret over i corona-taberne, som vil vinde ved en genåbning. 
Man må bare sige, at vi ikke har særlig mange corona-tabere herhjemme, og det nød vi godt af i 2020. 
Det er det, der koster på afkastet i 2021,” siger Jacob Pedersen. 
På listen over de værst ramte aktier finder man selskaber som Demant, Ambu, Ørsted og Vestas, som alle dykker over 5 pct. 
“Det er i særdeleshed de dyre og bæredygtige aktier, der bliver solgt ud af. 
Når Vestas og Ørsted er steget så kraftigt, som de er, så fylder de mere i det danske indeks, og så kan det altså mærkes, når de falder,” siger Jacob Pedersen. 
Liste over kurstab i C25-indekset C25 startede ellers ud med et lille plus, 
men siden har indekset bevæget sig i stik modsatte retning. Det er ikke kun i Danmark, at stemningen pludselig er blevet mere sur. 
“Vi har også set udviklingen i Europa er vendt fra at være neutral til at falde en smule nu,” siger Jacob Pedersen. 
Også de amerikanske aktiemarkeder er røde. Nasdaq-indekset falder 1,7 pct., mens S&P 500 er nede med 0,7 pct. Dow Jones ligger stille omkring nul.
"""
```

## Inference with the model
```python
def qa(question, context):
  inputs = tokenizer.encode_plus(question, context, return_tensors="pt") 
  answer_start_scores, answer_end_scores = model(**inputs)[0], model(**inputs)[1]
  answer_start = torch.argmax(answer_start_scores)
  answer_end = torch.argmax(answer_end_scores) + 1
  output = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
  return output

questions = ["Hvor mange danske eliteaktier er faldet mere end 3 pct",
             "Hvordan var det danske eliteindeks onsdag eftermiddag?",
             "Hvor meget faldte C25-indekset onsdag eftermiddag?",
             "Hvordan ser Nasdaq-indekset ud?",
             "Hvor meget falder Nasdaq-indekset?",
             "Hvilke selskaber har de værst ramte aktier?",
             'Hvad hedder aktieanalysechefen?'          
             ]

for q in questions:
  print('Spørgsmål: ',q)
  print('Svar: ',qa(q, context))

```
## Model output
```python
# Spørgsmål:  Hvor mange danske eliteaktier er faldet mere end 3 pct
# Svar:  11
# Spørgsmål:  Hvordan var det danske eliteindeks onsdag eftermiddag?
# Svar:  mørkerødt
# Spørgsmål:  Hvor meget faldte C25-indekset onsdag eftermiddag?
# Svar:  2, 5 pct
# Spørgsmål:  Hvordan ser Nasdaq-indekset ud?
# Svar:  værst
# Spørgsmål:  Hvor meget falder Nasdaq-indekset?
# Svar:  1, 7 pct., mens s [UNK] p 500 er nede med 0, 7 pct
# Spørgsmål:  Hvilke selskaber har de værst ramte aktier?
# Svar:  ambu, ørsted og vestas, som alle dykker over 5 pct. [UNK] det er i særdeleshed de dyre og bæredygtige aktier, der bliver solgt ud af. nar vestas og ørsted
# Spørgsmål:  Hvad hedder aktieanalysechefen?
# Svar:  jacob pedersen
```

### Dataset

* Pre-trained BERT was fine-tuned on the machine-translated [SQuADv1.1-da](https://github.com/ccasimiro88/TranslateAlignRetrieve/tree/multilingual/squads-tar/da).
* Example context credit goes to Dagbladet Børsen A/S.

### Contact

* For information on usage, fine-tuning procedure and more, please reach out on email through [jacobhein.com](https://jacobhein.com/#contact).

## Reference

* [Pre-trained model by BotXO](https://github.com/botxo/nordic_bert)

