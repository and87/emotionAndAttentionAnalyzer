# Emotion And Attention Analyzer

Questo repository usa:

- `EmotiEffLib` per il riconoscimento delle emozioni Ekman a 7 classi
- `OpenFace-3.0` per rilevamento volto e stima dello sguardo

## Ambiente Python consigliato

La base consigliata per questo progetto e' `Python 3.11`.

Motivazioni pratiche:

- `EmotiEffLib` non dichiara un vincolo esplicito su `requires-python`, ma il repository e' allineato a tooling moderno senza segnali specifici per `3.12/3.13`.
- `OpenFace-3.0` dichiara genericamente `Python 3.6+`, ma nel codice importa dipendenze come `torch`, `torchvision`, `timm`, `opencv` e `dlib`, che su Windows tendono a essere piu' affidabili su `3.11` rispetto a versioni piu' recenti.
- `Python 3.13` e' troppo nuovo per essere la scelta piu' prudente in una migrazione che deve integrare repository terzi non perfettamente puliti.

## Virtual Environment

Il virtual environment del progetto e' stato creato in `.venv` usando `Python 3.11`.

Per ricrearlo:

```powershell
& "C:\Users\and_g\AppData\Local\Programs\Python\Python311\python.exe" -m venv .venv
```

Per attivarlo in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Per verificare interprete e pip:

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip --version
```

## Dipendenze Installate

Per rendere l'ambiente ripetibile e' stato aggiunto [requirements.txt](/c:/workspace/emotionAndAttentionAnalyzer/requirements.txt:1).

Scelte importanti:

- `timm==0.9.16` e' stato scelto come compromesso compatibile con `EmotiEffLib`, che documenta il ramo `0.9.*`.
- Per OpenCV e' stato installato `opencv-contrib-python`, che copre anche gli use case del pacchetto base.
- `Pillow` usa un pin condizionale: `9.4.0` su Python < 3.12 e `12.2.0` su Python >= 3.12, per evitare build da sorgente senza header JPEG su Ubuntu 24.04.
- `EmotiEffLib` e' installata in editable mode dal clone locale.

Comando di installazione:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Per usare la GPU NVIDIA nel batch serve anche una build CUDA di PyTorch:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
```

## Pipeline Attuale

Il vecchio stack TensorFlow/TFLite/OpenCV-DNN/Keras e' stato rimosso per evitare ambiguita'. L'entrypoint principale e' [main.py](/c:/workspace/emotionAndAttentionAnalyzer/main.py:1), che usa [attention_analyzer.py](/c:/workspace/emotionAndAttentionAnalyzer/attention_analyzer.py:1).

La pipeline per ogni frame e':

- RetinaFace da `OpenFace-3.0` rileva i volti.
- Il modello multitask di `OpenFace-3.0` produce `gaze_yaw` e `gaze_pitch`.
- STAR da `OpenFace-3.0` produce i landmark WFLW/98 usati per `Ear` e posa della testa.
- `EmotiEffLib` con modello `enet_b2_7` produce le emozioni Ekman a 7 classi.
- `EmotiEffLib` con modello `enet_b0_8_va_mtl` produce `Arousal` e le feature usate dal classificatore temporale `Engaged`.
- I risultati sono salvati in CSV dentro `csv/`.

Uso:

```powershell
.\.venv\Scripts\python.exe main.py --video-root video --output-root csv --frame-step 2 --device cuda --require-gpu
```

Se `--device` viene omesso, viene usata automaticamente `cuda:0` se PyTorch vede CUDA, altrimenti `cpu`.
`--require-gpu` evita fallback silenziosi: se CUDA non e' disponibile, il batch si ferma. All'avvio `main.py`
stampa il device effettivo, per esempio `cuda:0 (NVIDIA GeForce ..., 4.0 GB)`.
Se rilanci il batch dopo un'interruzione, il resume e' automatico: i video con CSV gia' completi vengono saltati
e quelli con CSV parziale riprendono dal frame successivo. Per forzare la ripartenza da zero usare `--no-resume`.

## Demo Webcam

Per visualizzare la pipeline in tempo reale da webcam:

```powershell
.\.venv\Scripts\python.exe demo_webcam.py --device cpu
```

Comandi utili:

```powershell
.\.venv\Scripts\python.exe demo_webcam.py --camera 0 --width 1280 --height 720 --analyze-every 3
```

La finestra mostra il video live e aggiorna in overlay emozione principale, `Arousal`, `Engaged`, `Attention`, gaze OpenFace, posa testa ed `Ear`. L'inferenza gira in background: se il computer e' lento la webcam resta fluida, mentre i valori in overlay si aggiornano appena il modello termina un'elaborazione. L'overlay mostra anche `age_ms`, cioe' da quanti millisecondi risale l'ultima analisi. Premere `q` o `ESC` per uscire.

Il classificatore `Engaged` lavora su finestre temporali di feature facciali. Per default usa 128 frame analizzati:

```powershell
.\.venv\Scripts\python.exe main.py --engagement-window 128
```

Con `--frame-step 2` e video a 25 FPS, la prima finestra completa arriva dopo circa 10 secondi di video. Per evitare righe iniziali senza label, il primo valore `Engaged`/`EngagementLabel` calcolato viene replicato anche sulle righe precedenti usate per riempire la finestra. Se un video e' piu' corto della finestra, viene usata la finestra parziale disponibile a fine video.

## Output CSV

Le nuove colonne CSV sono:

```text
VideoTime;numOfPersons;Neutral;Happiness;Surprise;Sadness;Anger;Disgust;Fear;Arousal;Engaged;EngagementLabel;mainEmotion;OpenFaceGazeYawRad;OpenFaceGazePitchRad;OpenFaceGazeYawDeg;OpenFaceGazePitchDeg;OpenFaceGazeVectorX;OpenFaceGazeVectorY;OpenFaceGazeVectorZ;HeadYawDeg;HeadPitchDeg;HeadRollDeg;Ear
```

Le colonne legacy `Attention`, `lookingDirection`, `leftEyeHorizontalRatio`, `rightEyeHorizontalRatio`, `Valence` ed `Engagement` non vengono piu' scritte.

`Attention` resta calcolata internamente per la demo webcam, ma non viene piu' scritta nei CSV. E' un valore binario prudente basato su gaze, occhi aperti e posa testa:

- `Attention = 1` se `abs(OpenFaceGazeYawDeg) <= 20`, `abs(OpenFaceGazePitchDeg) <= 18`, `Ear >= 0.20`, `abs(HeadYawDeg) <= 35` e `abs(HeadPitchDeg) <= 25`.
- `Attention = 0` se almeno una delle soglie viene superata, se gli occhi risultano chiusi, o se `Ear` non e' affidabile.

`Ear` viene considerato non affidabile quando la posa e' troppo laterale (`abs(HeadYawDeg) > 40`) o quando il valore grezzo e' fuori scala (`raw_ear > 0.45`). Nella demo sono mostrati anche `raw_ear` e i valori sinistro/destro.

Le soglie sono configurabili da CLI:

```powershell
.\.venv\Scripts\python.exe demo_webcam.py --gaze-yaw-threshold-deg 20 --gaze-pitch-threshold-deg 18 --ear-attention-threshold 0.20 --head-yaw-attention-threshold-deg 35
```

`Arousal` e' l'output continuo del modello EmotiEffLib `enet_b0_8_va_mtl`. La libreria restituisce anche `Valence`, ma non viene scritto nel CSV per evitare ambiguita' con la vecchia euristica rimossa.

`Engaged` e' la probabilita' percentuale della classe `Engaged` stimata dal classificatore temporale originale di EmotiEffLib. `EngagementLabel` vale `Engaged` o `Distracted` in base alla classe con probabilita' maggiore.

`Ear` viene calcolato dai landmark STAR degli occhi WFLW/98. Per ogni occhio si usa:

```text
EAR = (|p1-p7| + |p2-p6| + |p3-p5|) / (3 * |p0-p4|)
```

Il valore scritto nel CSV e' la media tra occhio destro e sinistro.

`HeadYawDeg`, `HeadPitchDeg` e `HeadRollDeg` sono stimati dai landmark STAR con `cv2.solvePnP` e un modello 3D canonico del volto. Sono diversi da `OpenFaceGazeYawDeg` e `OpenFaceGazePitchDeg`: i primi descrivono la posa della testa, i secondi la direzione dello sguardo.

## Stato OpenFace-3.0

Le dipendenze Python principali del progetto sono installate e gli import di base funzionano.

Il clone locale di `OpenFace-3.0` usa due repository vendorizzati come gitlink:

- `OpenFace-3.0/STAR`: `https://github.com/ZhenglinZhou/STAR.git`, commit `9b125749b0d35766ed83d047036d1aa5e384984c`
- `OpenFace-3.0/Pytorch_Retinaface`: `https://github.com/biubug6/Pytorch_Retinaface.git`, commit `b984b4b775b2c4dced95c1eadd195a5c7d32a60b`

Sono stati popolati con i repository upstream compatibili e risultano allineati ai commit attesi dal tree di `OpenFace-3.0`.

Sono state applicate anche tre micro-patch di compatibilita':

- `OpenFace-3.0/STAR/demo.py`: `gradio` e' opzionale, per evitare una dipendenza web non necessaria alla pipeline.
- `OpenFace-3.0/STAR/lib/metric/fr_and_auc.py`: fallback da `scipy.integrate.simps` a `scipy.integrate.simpson` per SciPy recente.
- `OpenFace-3.0/Pytorch_Retinaface/models/retinaface.py`: il pretrain `mobilenetV1X0.25_pretrain.tar` viene risolto dalla cartella locale di OpenFace, non dalla working directory corrente.

Per importare il codice legacy di OpenFace senza gestire manualmente il `PYTHONPATH`, usare [openface3_runtime.py](/c:/workspace/emotionAndAttentionAnalyzer/openface3_runtime.py:1):

```python
from openface3_runtime import load_openface3_interface

openface3_interface = load_openface3_interface()
```

Smoke test attuale:

- `OpenFace-3.0/interface.py` viene importato correttamente.
- Il modello multitask `stage2_epoch_7_loss_1.1606_acc_0.5589.pth` viene caricato correttamente.
- RetinaFace carica correttamente `mobilenet0.25_Final.pth`.
- STAR carica `Landmark_98.pkl` per i landmark WFLW/98.
