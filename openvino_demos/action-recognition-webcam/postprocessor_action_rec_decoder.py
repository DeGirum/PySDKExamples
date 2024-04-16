import degirum as dg
import numpy as np
import json
class ActionRecDecoderPostprocessor(dg.postprocessor.ClassificationResults):
    CATEGORY_ID=0
    # labels = list(decoder_model.label_dictionary.values())
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs)
        new_inference_results = []
        raw_prediction = self._inference_results[0]["data"]
        probs = self._softmax(raw_prediction - np.max(raw_prediction))
        with open(self._model_params.LabelsPath, 'r') as json_file:
            action_labels = json.load(json_file)
        labels = list(action_labels.values())
        # Decodes top probabilities into corresponding label names
        decoded_labels, decoded_top_probs = self._decode_output(probs, labels, top_k=3)
        # print (decoded_labels, decoded_top_probs)
        for i in range(len(decoded_labels)):
            result = {
                "label": decoded_labels[i],
                "category_id": self.CATEGORY_ID,
                "score": decoded_top_probs[i],
            }
            new_inference_results.append(result)
        self._inference_results = new_inference_results

    def _softmax(self, x):
        """
        Normalizes logits to get confidence values along specified axis
        x: np.array, axis=None
        """
        exp = np.exp(x)
        return exp / np.sum(exp, axis=None)
    

    def _decode_output(self, probs, labels, top_k=3):
        """
        Decodes top probabilities into corresponding label names

        :param probs: confidence vector for 400 actions
        :param labels: list of actions
        :param top_k: The k most probable positions in the list of labels
        :returns: decoded_labels: The k most probable actions from the labels list
                decoded_top_probs: confidence for the k most probable actions
        """
        top_ind = np.argsort(-1 * probs)[:top_k]
        out_label = np.array(labels)[top_ind.astype(int)]
        decoded_labels = [out_label[0][0], out_label[0][1], out_label[0][2]]
        top_probs = np.array(probs)[0][top_ind.astype(int)]
        decoded_top_probs = [top_probs[0][0], top_probs[0][1], top_probs[0][2]]
        return decoded_labels, decoded_top_probs
