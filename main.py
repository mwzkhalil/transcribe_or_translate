from flask import Flask, request, jsonify
import os
from pathlib import Path
from faster_whisper import WhisperModel
import ctranslate2
import sentencepiece as spm
from time import perf_counter
import tempfile

# Initialize Flask app
app = Flask(__name__)

# Load models globally to avoid reloading on every request
model_medium_dir = "medium"
model_large_dir = "large-v3"
ct_model_path = "models/NLLB/nllb-200-distilled-600M-int8"
sp_model_path = "models/NLLB/flores200_sacrebleu_tokenizer_spm.model"

@app.route("/transcribe_or_translate", methods=["POST"])
def transcribe_or_translate():
    try:
        # Get the uploaded files and task parameters from the request
        files = request.files.getlist("audio_files")
        model_type = request.form.get("model", "medium")
        task = request.form.get("task", "transcribe")

        # Load the Whisper model based on user choice
        if model_type == "large":
            model = WhisperModel(model_large_dir, device="cpu", compute_type="int8")
        else:
            model = WhisperModel(model_medium_dir, device="cpu", compute_type="int8")

        # Initialize translation components if needed
        sp = None
        translator = None
        if task == "translate":
            sp = spm.SentencePieceProcessor()
            sp.load(sp_model_path)
            translator = ctranslate2.Translator(ct_model_path, device="cpu")

        # Process each file
        results = []
        for file in files:
            # Save the uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                file.save(tmp_file.name)

                # Run the Whisper model
                t1 = perf_counter()
                segments, info = model.transcribe(tmp_file.name, beam_size=5, task=task)

                result = {
                    "filename": file.filename,
                    "detected_language": info.language,
                    "language_probability": info.language_probability,
                    "transcription": [],
                    "translation": [] if task == "translate" else None
                }

                # Collect transcription/translation results
                for segment in segments:
                    result["transcription"].append({
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text
                    })

                    # Translate if the task is set to translation
                    if task == "translate":
                        translated_str = translate2(segment.text[1:], sp, translator)
                        result["translation"].append(translated_str)

                # Add the result to the list
                results.append(result)

        t2 = perf_counter()

        # Return the results as JSON
        return jsonify({
            "status": "success",
            "elapsed_time": round(t2 - t1, 2),
            "results": results
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# Helper function for translation (adapted from original script)
def translate2(source_sents, sp, translator):
    source_sentences = [source_sents]

    src_lang = "eng_Latn"
    tgt_lang = "urd_Arab"  # default output is Urdu
    beam_size = 5

    source_sentences = [sent.strip() for sent in source_sentences]
    target_prefix = [[tgt_lang]] * len(source_sentences)

    # Subword the source sentences
    source_sents_subworded = sp.encode_as_pieces(source_sentences)
    source_sents_subworded = [[src_lang] + sent + ["</s>"] for sent in source_sents_subworded]

    # Translate the source sentences
    translations_subworded = translator.translate_batch(source_sents_subworded, batch_type="tokens",
                                                        max_batch_size=2024, beam_size=beam_size,
                                                        target_prefix=target_prefix)
    translations_subworded = [translation.hypotheses[0] for translation in translations_subworded]
    for translation in translations_subworded:
        if tgt_lang in translation:
            translation.remove(tgt_lang)

    # Desubword the target sentences
    translations = sp.decode(translations_subworded)
    return translations


# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
