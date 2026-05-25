# VSA Pipeline — Plant Disease Recognition System

> **Progetto:** `plant-vsa-engine` — Motore neuro-simbolico VSA/HDC per diagnosi di malattie delle piante  
> **Linguaggio:** Java 11 + Maven  
> **Autore:** org.antonio  
> **Nota:** Il codice è commentato in italiano. I commenti originali sono stati preservati.

---

## Indice

1. [Modello CNN (Encoder)](#1-modello-cnn-encoder)
2. [Trasformazione CNN → BSC (Random Projection)](#2-trasformazione-cnn--bsc-random-projection)
3. [Operazione di Binding (XOR) — Bundling NON implementato](#3-operazione-di-binding-xor--bundling-non-implementato)
4. [Creazione degli Archetipi (Few-Shot Learning)](#4-creazione-degli-archetipi-few-shot-learning)
5. [Classificazione (Similarità e Z-Score)](#5-classificazione-similarità-e-z-score)
6. [Script Principale (Flusso Completo)](#6-script-principale-flusso-completo)
7. [Configurazione e Parametri](#7-configurazione-e-parametri)

---

## 1. Modello CNN (Encoder)

### 1.1 Interfaccia FeatureExtractor

**File:** `src/main/java/org/antonio/perception/FeatureExtractor.java`

```java
package org.antonio.perception;

public interface FeatureExtractor {
    /**
     * Prende un'immagine e restituisce le feature latenti o le probabilità.
     * @param imagePixels Array dei pixel dell'immagine
     * @return Array di float estratti dal modello TensorFlow Lite
     */
    float[] extractFeatures(float[][][] imagePixels);
}
```

### 1.2 Implementazione PC: PcFeatureExtractor (MobileNetV2 via TensorFlow SavedModel)

**File:** `src/main/java/org/antonio/perception/PcFeatureExtractor.java`

Questo è l'estrattore **attivo** per architettura x86 (PC). Carica un modello MobileNetV2 in formato TensorFlow SavedModel, auto-scopre le firme di input/output, ed estrae **1280 feature** latenti.

```java
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
```

### 1.3 Implementazione Android (STUB — commentata)

**File:** `src/main/java/org/antonio/perception/MobileNetFeatureExtractor.java`

Versione per TensorFlow Lite su Android. Tutto il codice è commentato (placeholder). L'interfaccia `FeatureExtractor` NON è implementata (manca `implements`).

```java
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
```

### 1.4 Preprocessing dell'Input (Caricamento + Normalizzazione MobileNetV2)

**File:** `src/main/java/org/antonio/Main.java` — metodo `loadImage()`

```java
/**
 * Legge un'immagine dal disco, la ridimensiona a 224x224 e la normalizza
 * nel formato richiesto da MobileNetV2 (valori tra -1.0 e 1.0).
 */
public static float[][][] loadImage(String imagePath) throws Exception {
    File file = new File(imagePath);
    if (!file.exists()) {
        throw new RuntimeException("Immagine non trovata: " + file.getAbsolutePath());
    }

    BufferedImage originalImg = ImageIO.read(file);

    // Ridimensionamento forzato a 224x224 pixel
    BufferedImage resizedImg = new BufferedImage(224, 224, BufferedImage.TYPE_INT_RGB);
    Graphics2D g = resizedImg.createGraphics();
    g.drawImage(originalImg, 0, 0, 224, 224, null);
    g.dispose();

    // Estrazione RGB e normalizzazione Keras
    float[][][] tensor = new float[224][224][3];
    for (int y = 0; y < 224; y++) {
        for (int x = 0; x < 224; x++) {
            int rgb = resizedImg.getRGB(x, y);
            float r = ((rgb >> 16) & 0xFF);
            float g_color = ((rgb >> 8) & 0xFF);
            float b = (rgb & 0xFF);

            // Formula di Keras per MobileNetV2: (valore / 127.5) - 1.0
            tensor[y][x][0] = (r / 127.5f) - 1.0f;
            tensor[y][x][1] = (g_color / 127.5f) - 1.0f;
            tensor[y][x][2] = (b / 127.5f) - 1.0f;
        }
    }
    return tensor;
}
```

### 1.5 Modello Salvato

**Directory:** `src/main/resources/mobilenet_savedmodel/`

```
mobilenet_savedmodel/
├── fingerprint.pb
├── saved_model.pb
└── variables/
    ├── variables.data-00000-of-00001
    └── variables.index
```

- **Architettura:** MobileNetV2 (feature extractor, non classifier — output a 1280 dimensioni, penultimo layer prima della testa di classificazione)
- **Input:** Batch `[1, 224, 224, 3]` di float normalizzati in `[-1.0, 1.0]`
- **Output:** `[1, 1280]` feature latenti

---

## 2. Trasformazione CNN → BSC (Random Projection)

**File:** `src/main/java/org/antonio/bridge/RandomProjectionBridge.java`

Questo è il "ponte neuro-simbolico": proietta il vettore continuo `float[1280]` della CNN in un ipervettore BSC (Binary Spatter Code) di **10.000 dimensioni** tramite:

1. **Matrice di proiezione casuale fissa** (seed deterministico) di dimensioni `[1280 × 10000]` con valori `+1` / `-1` (memorizzata come `boolean[][]` per risparmiare RAM: `true = +1`, `false = -1`)
2. **Prodotto scalare** tra ogni feature CNN e la colonna corrispondente della matrice
3. **Soglia (sign function):** `sum > 0 → bit = 1`, altrimenti `bit = 0`
4. **Bit-packing:** ogni 64 bit vengono impacchettati in un `long`

```java
package org.antonio.bridge;
import org.antonio.vsa.VsaVector;

import java.util.Random;

/**
 * Traduce i float in uscita dalla CNN in bit per la VSA (SBC).
 */
public class RandomProjectionBridge {

    private final int cnnFeatureSize;
    private final int vsaDimensions;

    // Invece di float, usiamo boolean per la matrice per risparmiare RAM!
    // true = +1, false = -1
    private final boolean[][] projectionMatrix;

    /**
     * Inizializza il ponte.
     * @param cnnFeatureSize Es. 512 (Test B) o 38 (Test A)
     * @param seed Il seme fisso per garantire che la matrice sia sempre identica
     */
    public RandomProjectionBridge(int cnnFeatureSize, long seed) {
        this.cnnFeatureSize = cnnFeatureSize;
        this.vsaDimensions = VsaVector.DIMENSIONS;
        this.projectionMatrix = new boolean[cnnFeatureSize][vsaDimensions];

        // Inizializziamo il generatore con il nostro seme fisso
        Random random = new Random(seed);

        // Riempiamo la matrice di +1 (true) e -1 (false)
        for (int i = 0; i < cnnFeatureSize; i++) {
            for (int d = 0; d < vsaDimensions; d++) {
                projectionMatrix[i][d] = random.nextBoolean();
            }
        }
    }

    /**
     * Il momento magico: da array continuo (CNN) a Ipervettore BSC
     */
    public VsaVector projectToVsa(float[] cnnFeatures) {
        if (cnnFeatures.length != cnnFeatureSize) {
            throw new IllegalArgumentException("Le feature CNN non combaciano con la matrice!");
        }

        // Calcoliamo quanti 'long' ci servono per contenere 10000 bit (157)
        int numBlocks = (vsaDimensions + 63) / 64;
        long[] bscBlocks = new long[numBlocks];

        // Iteriamo su ogni dimensione del nuovo ipervettore (da 0 a 9999)
        for (int d = 0; d < vsaDimensions; d++) {

            // 1. Calcoliamo il Prodotto Scalare (Dot Product)
            float sum = 0.0f;
            for (int i = 0; i < cnnFeatureSize; i++) {
                // Se la matrice in quella cella è true (+1), sommiamo la feature.
                // Se è false (-1), la sottraiamo.
                if (projectionMatrix[i][d]) {
                    sum += cnnFeatures[i];
                } else {
                    sum -= cnnFeatures[i];
                }
            }

            // 2. La Funzione Segno (Thresholding)
            // Se la somma è > 0, decidiamo che il bit sarà 1. Se <= 0, sarà 0.
            if (sum > 0) {
                // 3. Bit-Packing: Inseriamo l'1 nel posto giusto dentro l'array di long!
                int blockIndex = d / 64; // Trova in quale dei 157 long dobbiamo andare
                int bitPosition = d % 64; // Trova quale dei 64 bit di QUEL long dobbiamo accendere

                // Usiamo l'operatore di Shift (<<) e l'operatore OR (|)
                bscBlocks[blockIndex] |= (1L << bitPosition);
            }
            // Se sum <= 0, non facciamo nulla, perché gli array in Java nascono già pieni di zeri.
        }

        return new VsaVector(bscBlocks);
    }
}
```

---

## 3. Operazione di Binding (XOR) — Bundling NON implementato

**File:** `src/main/java/org/antonio/vsa/VsaVector.java`

Il progetto implementa il **binding** (associazione) tramite XOR bit-a-bit su blocchi da 64 bit.  
**NON è presente una funzione di bundling** (superposizione/somma con voto a maggioranza). Il sistema fa few-shot learning con **un solo esempio per classe**, memorizzando direttamente il vettore proiettato come archetipo (anziché fare il bundle di molteplici esempi).  
La similarità è calcolata come **distanza di Hamming** e convertita in **similarità del coseno**.

```java
package org.antonio.vsa;

/**
 * Rappresenta un Ipervettore Bipolare (MAP-B) compresso in un array di bit (long).
 */
public class VsaVector {

    // 10000 dimensioni divise per 64 bit = 157 long
    // (l'ultimo array avrà dei bit inutilizzati, ma va bene)
    public static final int DIMENSIONS = 10000;
    private static final int ARRAY_SIZE = (DIMENSIONS + 63) / 64;

    private final long[] blocks;

    /**
     * Costruisce un vettore BSC a partire da un array di blocchi a 64 bit.
     */
    public VsaVector(long[] blocks) {
        if (blocks.length != ARRAY_SIZE) {
            throw new IllegalArgumentException("Dimensione array errata");
        }
        this.blocks = blocks;
    }

    /**
     * OPERAZIONE 1: BINDING (Associazione)
     * In BSC, il binding è l'operatore logico XOR.
     */
    public VsaVector bind(VsaVector other) {
        long[] result = new long[ARRAY_SIZE];
        for (int i = 0; i < ARRAY_SIZE; i++) {
            // Un singolo ciclo di clock del processore esegue il binding di 64 dimensioni!
            result[i] = this.blocks[i] ^ other.blocks[i];
        }
        return new VsaVector(result);
    }

    /**
     * LA MEMORIA CLEAN-UP: Distanza di Hamming
     * Calcola quanti bit sono diversi tra questo vettore e un altro.
     */
    public int calculateHammingDistance(VsaVector other) {
        int distance = 0;
        for (int i = 0; i < ARRAY_SIZE; i++) {
            // Troviamo i bit discordanti con lo XOR
            long differences = this.blocks[i] ^ other.blocks[i];

            // MAGIA HARDWARE: POPCNT
            // Il processore (ARM o x86) conta i bit a '1' a livello di silicio.
            distance += Long.bitCount(differences);
        }
        return distance;
    }

    /**
     * Ritorna la Similarità del Coseno equivalente (utile per paragoni statistici e Z-Score).
     * Formula: 1 - 2 * (Hamming / Dimensioni)
     */
    public double getCosineSimilarity(VsaVector other) {
        int h = calculateHammingDistance(other);
        return 1.0 - 2.0 * ((double) h / DIMENSIONS);
    }

    public long[] getBlocks() {
        return blocks;
    }
}
```

**Dettagli implementativi:**

| Caratteristica | Valore |
|---|---|
| Dimensione ipervettore | **10.000 bit** |
| Rappresentazione interna | `long[]` di 157 elementi (bit-packing: 64 bit per long) |
| Binding | XOR (`^`) su `long`, processa 64 dimensioni per ciclo CPU |
| Hamming distance | XOR + `Long.bitCount()` (istruzione POPCNT hardware) |
| Cosine similarity | `1 − 2 × (hammingDistance / DIMENSIONS)` |
| Bundling (voto a maggioranza) | **NON implementato** |

---

## 4. Creazione degli Archetipi (Few-Shot Learning)

**File:** `src/main/java/org/antonio/domain/PlantDiseaseDiagnostician.java` — metodi `teachHealthy()` e `teachSick()`

Il sistema usa **few-shot learning con 1 esempio per classe** (one-shot). Non c'è bundling di molteplici vettori: ogni archetipo è il singolo vettore BSC ottenuto dalla proiezione dell'immagine di training.

```java
/**
 * FEW-SHOT LEARNING: Insegna al sistema cos'è una "Pianta Sana"
 * L'utente scatta una foto di una foglia sana e la passa qui.
 */
public void teachHealthy(float[][][] image) {
    float[] features = cnn.extractFeatures(image);
    this.healthyReference = bridge.projectToVsa(features);
    System.out.println("Vettore 'Sana' memorizzato con successo!");
}

/**
 * FEW-SHOT LEARNING: Insegna al sistema cos'è la "Malattia"
 * L'utente scatta una foto di una foglia malata e la passa qui.
 */
public void teachSick(float[][][] image) {
    float[] features = cnn.extractFeatures(image);
    this.sickReference = bridge.projectToVsa(features);
    System.out.println("Vettore 'Malata' memorizzato con successo!");
}
```

**Pipeline di creazione archetipo:**
1. `cnn.extractFeatures(image)` → `float[1280]`
2. `bridge.projectToVsa(features)` → `VsaVector` (10.000 bit)
3. Il vettore viene memorizzato come `healthyReference` o `sickReference`

**Nota:** Non esiste logica per raccogliere vettori da un dataset (es. PlantVillage) e fare il bundle. Il sistema parte come "tabula rasa" (`healthyReference = null`, `sickReference = null`) e impara con esattamente 1 esempio per classe.

---

## 5. Classificazione (Similarità e Z-Score)

**File:** `src/main/java/org/antonio/domain/PlantDiseaseDiagnostician.java` — metodo `diagnose()`

La classificazione confronta il vettore query con entrambi gli archetipi usando la **similarità del coseno** derivata dalla distanza di Hamming, e applica una soglia statistica basata sullo **Z-Score** (3 sigma ≈ 99.7% confidenza).

```java
public String diagnose(float[][][] image) {
    if (healthyReference == null || sickReference == null) {
        throw new IllegalStateException("Devi prima addestrare il sistema!");
    }

    float[] features = cnn.extractFeatures(image);
    VsaVector queryVector = bridge.projectToVsa(features);

    double simHealthy = queryVector.getCosineSimilarity(healthyReference);
    double simSick = queryVector.getCosineSimilarity(sickReference);

    // Calcolo dello Z-Score
    double sigma = 1.0 / Math.sqrt(VsaVector.DIMENSIONS); // 0.01 per D=10000
    double zScoreHealthy = simHealthy / sigma;
    double zScoreSick = simSick / sigma;

    System.out.println("   --- ANALISI MATEMATICA VSA ---");
    System.out.printf("   Distanza da [Pianta Sana]   : %.4f (Z-Score: %.1f Sigma)%n", simHealthy, zScoreHealthy);
    System.out.printf("   Distanza da [Pianta Malata] : %.4f (Z-Score: %.1f Sigma)%n", simSick, zScoreSick);

    // Definiamo una soglia minima di certezza (es. 3 Sigma, ~99.7% di confidenza)
    if (Math.max(zScoreHealthy, zScoreSick) < 3.0) {
        return "Sconosciuta (Troppo rumore, la rete non è sicura!)";
    }

    return simSick > simHealthy ? "Malata!" : "Sana!";
}
```

**Metrica di similarità e classificazione:**

| Componente | Dettaglio |
|---|---|
| Similarità usata | **Similarità del coseno** derivata da Hamming: `cos = 1 − 2 × (Hamming / 10000)` |
| Range similarità | `[-1, +1]` (identici → +1, ortogonali → 0, opposti → −1) |
| Deviazione standard teorica | `σ = 1 / √10000 = 0.01` (per vettori BSC casuali indipendenti) |
| Soglia di confidenza | **3 sigma** (Z-Score ≥ 3.0, ~99.7% confidenza) |
| Classificazione finale | `argmax(simHealthy, simSick)` con controllo soglia minima |
| Output incerto | `"Sconosciuta (Troppo rumore, la rete non è sicura!)"` se nessuno Z-Score raggiunge 3σ |

---

## 6. Script Principale (Flusso Completo)

**File:** `src/main/java/org/antonio/Main.java`

Questo è l'unico script che mostra il flusso end-to-end. **Non esistono test JUnit** (la dipendenza `junit-jupiter-engine` è dichiarata nel POM ma non ci sono file di test).

```java
package org.antonio;

import org.antonio.bridge.RandomProjectionBridge;
import org.antonio.domain.PlantDiseaseDiagnostician;
import org.antonio.perception.FeatureExtractor;
import org.antonio.perception.PcFeatureExtractor;

import javax.imageio.ImageIO;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;

public class Main {

    public static void main(String[] args) {
        System.out.println("--- AVVIO MOTORE VSA/HDC SU PC ---");

        // Assicurati che questo percorso punti alla cartella estratta da Colab
        String modelPath = "src/main/resources/mobilenet_savedmodel";
        FeatureExtractor realPcEye = null;

        try {
            // 1. INIZIALIZZAZIONE DEL SISTEMA NEURO-SIMBOLICO
            realPcEye = new PcFeatureExtractor(modelPath);
            RandomProjectionBridge bridge = new RandomProjectionBridge(1280, 42L); // 42 è il nostro seme fisso
            PlantDiseaseDiagnostician doctor = new PlantDiseaseDiagnostician(realPcEye, bridge);

            // 2. CARICAMENTO DELLE FOTO REALI
            System.out.println("\nCaricamento immagini dal disco in corso...");
            float[][][] imageHealthy = loadImage("sana.jpg");
            float[][][] imageSick = loadImage("malata.jpg");
            float[][][] imageTest = loadImage("test.jpg");
            System.out.println("Immagini caricate e processate con successo.");

            // 3. FEW-SHOT LEARNING (Apprendimento sul campo)
            System.out.println("\n[Fase 1: Few-Shot Learning]");
            doctor.teachHealthy(imageHealthy);
            doctor.teachSick(imageSick);

            // 4. DIAGNOSI DELL'IMMAGINE INCOGNITA
            System.out.println("\n[Fase 2: Diagnosi della nuova foto]");
            String diagnosis = doctor.diagnose(imageTest);

            System.out.println("\n*** VERDETTO FINALE: " + diagnosis + " ***");

        } catch (Exception e) {
            System.err.println("Errore durante l'esecuzione:");
            e.printStackTrace();
        } finally {
            // 5. PULIZIA DELLA MEMORIA
            if (realPcEye instanceof PcFeatureExtractor) {
                ((PcFeatureExtractor) realPcEye).close();
                System.out.println("\nMotore TensorFlow chiuso correttamente.");
            }
        }
    }

    // ... loadImage() già mostrato nella sezione 1.4
}
```

**Immagini di riferimento nella root del progetto:**

| File | Ruolo |
|---|---|
| `sana.jpg` | Foglia sana per few-shot learning (classe "healthy") |
| `malata.jpg` | Foglia malata per few-shot learning (classe "sick") |
| `malata_1.jpg` | Seconda immagine malata (NON usata da `Main.java`) |
| `test.jpg` | Immagine di test per la diagnosi |

---

## 7. Configurazione e Parametri

### 7.1 Maven POM

**File:** `pom.xml`

```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <groupId>org.antonio</groupId>
    <artifactId>plant-vsa-engine</artifactId>
    <version>1.0-SNAPSHOT</version>

    <properties>
        <maven.compiler.source>11</maven.compiler.source>   <!-- Java 11 -->
        <maven.compiler.target>11</maven.compiler.target>   <!-- Java 11 -->
        <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
    </properties>

    <dependencies>
        <!-- Motore di Test (JUnit 5) per provare la matematica della VSA su PC -->
        <dependency>
            <groupId>org.junit.jupiter</groupId>
            <artifactId>junit-jupiter-engine</artifactId>
            <version>5.9.2</version>
            <scope>test</scope>
        </dependency>
        <!-- TensorFlow Ufficiale per Java (Supporta Windows/Mac/Linux x86 e x64) -->
        <dependency>
            <groupId>org.tensorflow</groupId>
            <artifactId>tensorflow-core-platform</artifactId>
            <version>0.5.0</version>
        </dependency>
    </dependencies>
</project>
```

### 7.2 Parametri chiave (hardcoded nel codice)

| Parametro | Valore | File | Riga |
|---|---|---|---|
| **Dimensione vettore BSC** | `10000` | `VsaVector.java` | `9` |
| **Feature CNN in input** | `1280` | `PcFeatureExtractor.java` | `56` |
| **Seme matrice di proiezione** | `42L` | `Main.java` | `25` |
| **Dimensione immagine CNN** | `224 × 224 × 3` | `Main.java`, `PcFeatureExtractor.java` | — |
| **Normalizzazione** | `(pixel / 127.5) − 1.0` (MobileNetV2 style) | `Main.java` | `87-89` |
| **Soglia Z-Score** | `3.0` sigma (~99.7%) | `PlantDiseaseDiagnostician.java` | `66` |
| **Modello CNN** | `mobilenet_savedmodel` (SavedModel) | `Main.java` | `19` |
| **Java version** | 11 | `pom.xml` | `12-13` |

### 7.3 `.gitignore`

**File:** `.gitignore` — ignora `target/`, file IDE, `.DS_Store`. Nessuna configurazione rilevante per la VSA.

---

## Riepilogo Architetturale

```
┌─────────────────────────────────────────────────────────┐
│  Immagine (224×224×3 RGB)                               │
│       ↓ loadImage() + normalizzazione [-1, +1]          │
│  PcFeatureExtractor (MobileNetV2 SavedModel)            │
│       ↓ output: float[1280]                             │
│  RandomProjectionBridge.projectToVsa()                  │
│       ↓ Matrice casuale fissa [1280×10000] + segno      │
│       ↓ Bit-packing → long[157]                         │
│  VsaVector (10.000 bit BSC)                             │
│       ↓                                                 │
│  ┌──────────────────────────────────────────┐           │
│  │  Few-Shot Learning (1 esempio/classe)    │           │
│  │  healthyReference ← sana.jpg             │           │
│  │  sickReference    ← malata.jpg            │           │
│  └──────────────────────────────────────────┘           │
│       ↓                                                 │
│  PlantDiseaseDiagnostician.diagnose()                   │
│       ↓ Cosine similarity vs entrambi gli archetipi     │
│       ↓ Z-Score ≥ 3σ? → "Sana!" / "Malata!" / "?"      │
│  Verdetto finale                                        │
└─────────────────────────────────────────────────────────┘
```

---

## File Non Esistenti / Limitazioni

- **Nessun file `.yml`, `.yaml`, `.json`, `.env`, `.cfg`, `.toml`** di configurazione.
- **Nessun test JUnit** (la directory `src/test/` non esiste).
- **Nessuna logica di bundling** (superposizione con voto a maggioranza) — solo binding (XOR).
- **Nessun caricamento da dataset** (es. PlantVillage): il sistema impara da 1 immagine per classe caricata direttamente dal filesystem.
- **Nessun supporto multi-classe** oltre a sano/malato (classificazione binaria).
- **MobileNetFeatureExtractor** è completamente commentato (placeholder per Android TFLite).
- **Nessuna persistenza** degli archetipi: vengono persi al termine dell'esecuzione.
