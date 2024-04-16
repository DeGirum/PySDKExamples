import degirum as dg
import numpy as np

class OCRTextRecPostprocessor(dg.postprocessor.ClassificationResults):
    END_OF_STRING_SYMBOL = "~"
    TEXT_CATEGORY_ID = 0
    CONFIDENCE_SCORE = 1.0
    LETTERS = "~0123456789abcdefghijklmnopqrstuvwxyz"

    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs)
        raw_prediction = self._inference_results[0]["data"]
        squeezed_prediction = np.squeeze(raw_prediction)
        recognized_text = self._parse_recognition_results(squeezed_prediction)
        result = {
            "label": recognized_text,
            "category_id": self.TEXT_CATEGORY_ID,
            "score": self.CONFIDENCE_SCORE,
        }
        self._inference_results = [result]

    def _parse_recognition_results(self, recognition_results):
        annotation = []
        for letter_probabilities in recognition_results:
            parsed_letter = self.LETTERS[letter_probabilities.argmax()]
            if parsed_letter == self.END_OF_STRING_SYMBOL:
                break
            annotation.append(parsed_letter)
        return "".join(annotation)
