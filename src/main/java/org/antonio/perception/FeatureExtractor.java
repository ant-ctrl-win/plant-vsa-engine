package org.antonio.perception;

public interface FeatureExtractor {
    /**
     * Prende un'immagine e restituisce le feature latenti o le probabilità.
     * @param imagePixels Array dei pixel dell'immagine
     * @return Array di float estratti dal modello TensorFlow Lite
     */
    float[] extractFeatures(float[][][] imagePixels);
}