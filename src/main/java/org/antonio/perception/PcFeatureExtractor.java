package org.antonio.perception;

import org.tensorflow.SavedModelBundle;
import org.tensorflow.SessionFunction;
import org.tensorflow.Signature;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.types.TFloat32;
import java.util.Collections;

public class PcFeatureExtractor implements FeatureExtractor {

    private final SavedModelBundle model;
    private final String inputName;
    private final String outputName;

    public PcFeatureExtractor(String modelFolderPath) {
        System.out.println("Caricamento TensorFlow per x86 in corso...");
        this.model = SavedModelBundle.load(modelFolderPath, "serve");

        // --- L'AUTO-SCOPERTA DELLE FIRME (Auto-Discovery) ---
        // Estraiamo la funzione principale del modello
        SessionFunction function = this.model.function("serving_default");
        Signature signature = function.signature();

        // Chiediamo dinamicamente il nome esatto dell'ingresso e dell'uscita
        this.inputName = signature.inputNames().iterator().next();
        this.outputName = signature.outputNames().iterator().next();

        System.out.println("-> Porta di Input rilevata: " + this.inputName);
        System.out.println("-> Porta di Output rilevata: " + this.outputName);
        System.out.println("Modello caricato con successo!");
    }

    @Override
    public float[] extractFeatures(float[][][] imagePixels) {
        // 1. Creiamo il tensore vuoto
        TFloat32 imageTensor = TFloat32.tensorOf(Shape.of(1, 224, 224, 3));

        // --- IL FIX (Togliamo il tappo dall'obiettivo!) ---
        // Copiamo i veri pixel dell'immagine dentro il Tensore
        for (int y = 0; y < 224; y++) {
            for (int x = 0; x < 224; x++) {
                imageTensor.setFloat(imagePixels[y][x][0], 0, y, x, 0); // Canale R
                imageTensor.setFloat(imagePixels[y][x][1], 0, y, x, 1); // Canale G
                imageTensor.setFloat(imagePixels[y][x][2], 0, y, x, 2); // Canale B
            }
        }

        // 2. Passiamo i nomi rilevati dinamicamente
        Tensor outputTensor = model.call(Collections.singletonMap(this.inputName, imageTensor))
                .get(this.outputName)
                .get();

        // 3. Estraiamo le feature finali
        float[] features = new float[1280];
        ((TFloat32) outputTensor).asRawTensor().data().asFloats().read(features);

        imageTensor.close();
        outputTensor.close();

        return features;
    }

    public void close() {
        if (model != null) {
            model.close();
        }
    }
}