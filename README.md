**Simple API for Speech2Text using Faster-Whisper and optionally translation using CTranslate2 / NLLB**
![](/Demo.png?raw=true)

## Setup
```
pip install -r requirements.txt
```
Tested with WinPython 3.11


## Download models
**1. NLLB**<br>
https://pretrained-nmt-models.s3.us-west-2.amazonaws.com/CTranslate2/nllb/nllb-200_600M_int8_ct2.zip<br>
- unzip to "models/NLLB"<br>

**2. WHISPER**<br>
Download "config.json", "model.bin", "tokenizer.json" and "vocabulary.txt" from:<br>
https://huggingface.co/Systran/faster-whisper-medium/tree/main<br>
and<br>
https://huggingface.co/Systran/faster-whisper-large-v2/tree/main<br>

- place the medium model files inside "models/whisper_medium"
- place the large model files inside "models/whisper_large-3"

## Default Translation (target language)
```
Line 94: tgt_lang = "urd_Arab"
```
- default is Urdu: "urd_Arab"
- List of language codes, see: https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200

## References<br>
- Faster-Whisper: https://github.com/SYSTRAN/faster-whisper
- NLLB 200 with CTranslate2: https://forum.opennmt.net/t/nllb-200-with-ctranslate2/5090
