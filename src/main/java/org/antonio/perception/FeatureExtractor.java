package org.antonio.perception;

public interface FeatureExtractor {
    int CNN_FEATURE_SIZE = 1280;

    /**
     * Prende un'immagine e restituisce le feature latenti o le probabilità.
     * @param imagePixels Array dei pixel dell'immagine
     * @return Array di float estratti dal modello TensorFlow Lite
     */
    float[] extractFeatures(float[][][] imagePixels);
}