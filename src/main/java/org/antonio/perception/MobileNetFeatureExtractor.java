package org.antonio.perception;

//import org.tensorflow.lite.Interpreter;
import java.io.File;
import java.net.URL;

public class MobileNetFeatureExtractor { //implements FeatureExtractor {

//    private Interpreter tflite;
//
//    public MobileNetFeatureExtractor() {
//        try {
//            // Cerca il modello nella cartella resources
//            URL modelUrl = getClass().getClassLoader().getResource("mobilenet_v2_feature_extractor.tflite");
//            if (modelUrl == null) {
//                throw new RuntimeException("Modello TFLite non trovato in resources!");
//            }
//
//            // Inizializza il "cervello visivo" di TensorFlow
//            File modelFile = new File(modelUrl.toURI());
//            this.tflite = new Interpreter(modelFile);
//            System.out.println("Modello TensorFlow Lite caricato con successo!");
//
//        } catch (Exception e) {
//            e.printStackTrace();
//            throw new RuntimeException("Errore nel caricamento del modello.");
//        }
//    }
//
//    @Override
//    public float[] extractFeatures(float[][][] imagePixels) {
//        // TensorFlow Lite in Java si aspetta i dati in array multidimensionali.
//        // Input: [1 immagine][224 altezza][224 larghezza][3 canali RGB]
//        float[][][][] inputBatch = new float[1][224][224][3];
//        inputBatch[0] = imagePixels;
//
//        // Output: [1 immagine][1280 feature]
//        float[][] outputBatch = new float[1][1280];
//
//        // Questa è la riga che esegue la Rete Neurale!
//        tflite.run(inputBatch, outputBatch);
//
//        // Estraiamo il vettore 1D (le 1280 feature) dall'output
//        return outputBatch[0];
//    }
//
//    // È buona norma chiudere l'interprete quando si spegne l'app
//    public void close() {
//        if (tflite != null) {
//            tflite.close();
//        }
//    }
}