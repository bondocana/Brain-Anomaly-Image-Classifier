# Brain-Anomaly-Image-Classifier

Scopul acestui proiect este antrenarea unui model pentru a clasifica imagini în două clase de scanări ale creierului: una care conține anomalii(eticheta 1) și
una normală(eticheta 0).

Modele testate:

- Naive-Bayes

Am folosit modelul Multinomial Naive Bayes (MNB) din libraria sklearn.
Modelul presupune că toate caracteristicile sunt independente și urmează o
anumită distribuție de probabilitate, numită distribuție multinomială.

Am segmentat datele in 3 categorii, fiecare reprezentând un interval de pixeli. Acest lucru a optimizat modelul și a dus la o rezultate mai bune.

Accuracy: 0.722

f1_score: 0.40598290598290604

![image](https://user-images.githubusercontent.com/95356241/236637244-6d51530f-a77f-47e1-b8b3-8ff31024baa1.png)



- Convolutional Neural Network(CNN)

O Rețea Neuronală Convoluțională (CNN) este compusă din mai multe straturi, inclusiv straturi de convoluție, de max-pooling și de învățare a caracteristicilor. Straturile de convoluție scanează imaginea și detectează caracteristici precum marginile, colțurile, texturile etc.

La compilarea modelului am folosit funcția binary_crossentropy, optimizer-ul
adam folosit pentru a ajusta ponderile modelului în timpul antrenării și ca
metrică acuratețea.
Am antrenat modelul in 20 de epoci, pe datele de antrenare, cu batch_size de
64 și am obținut următoarele rezultate:

Loss: 0.4829079508781433

Accuracy: 0.9004999995231628

f1_score: 0.5446224256292906

![image](https://user-images.githubusercontent.com/95356241/236637257-fd73e9be-41a5-4d6e-9a1c-bc35260efe8f.png)

![image](https://user-images.githubusercontent.com/95356241/236637268-442a1082-760a-4d8d-993c-623683084bb9.png)

