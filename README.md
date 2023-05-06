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
![image](https://user-images.githubusercontent.com/95356241/236634665-e48bd04e-8c0f-4838-895d-c4f7a8c1c7cb.png)

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
![image](https://user-images.githubusercontent.com/95356241/236637185-78196d48-868c-4d6a-962b-79f2b6aa3ab8.png)
![image](https://user-images.githubusercontent.com/95356241/236637192-51fd1f37-7f79-4e60-90cc-76c2457fb54e.png)
