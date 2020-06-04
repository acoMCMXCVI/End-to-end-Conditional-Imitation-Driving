# End-to-end Conditional Imitation Driving 
Autonomous driving research (Part I)

![](Pictures/base.jpg)


Cilj projekta je bio kreirati sitem koji omogućava prikupljanje podataka I pripremu _dataset-a_, kao i treniranje neuronske mreže za _end-to-end imitation learning_ za autonomnu voznju, u svrsi istraživanja koncepata autonomne vožnje.

Sistem je kreiran za _Carla_ simulator.


## Video

Video na linku ispod prikazuje kratak demo dobijenih rezultata, kao I kratak prikaz značaja augmentacije podataka, te poređenje dva modela u specifičnim situacijama, jednog obučenog na _raw_ podacima, I jednog na kom je primenjena augmentacija. 

https://www.youtube.com/watch?v=LoXPs6NShLI

## Uputstvo

### Bitlioteke

```
Python: 3.7.4
Carla simulator: 0.9.8
Keras: 2.2.4
Tensorflow: 1.14.0
Imgaug: 0.4.0
OpenCV: 3.4.7
```

## Koraci

Ceo proces od prikupljanja podataka do treniranja mreže je podeljen u više koraka: 

1. __Prikupljanje podataka ```colecting_data.py```:__  
Prikupljanje podataka se sastoji iz preuzimanja slike sa kamere na automobilu koja se potom sa odgovarajućim uglom upravljanja smešta u listu. Prikupljeni podaci se čuvaju u _batch-evima_ od po 256 podatka x∈{_image, steering angle_}. Prilikom prikupljanja podataka, _Carla_ autopilot upravlja automobilom. Podatke je moguće prikupljati tokom više sesija.

2. __Pregled prikupljenih podataka ```show_colected_data.py```:__  
Pregled _raw_ podataka kroz sve _batch-eve_ jedne sesije.

3. __Pridodavanje _high-level_ kontrole podacima ```add_high_level_control.py```:__  
Nakon što prikupimo podatke, potrebno im je pridodati informaciju o _high-level_ kontroli, odnosno pridodati informaciju koja je navela "auto" na odgovarajuću akciju. Na početku je bitno definisati za koje se sve sesije želi proci kroz ovaj postupak. _High-level_ kontrola se dodaje tako što se na dugme ‘O’ na tastaturi bira početni frejm od koga važi određena _high-level_ kontrola, dok se na dugme ‘P’ na tastaturi bira krajnji frejm do kojeg je ta kontrola delovala, nakon čega se upisuje određena kontrola. Nakon prolaska kroz sve _batch-eve_ svih sesija, program čuva fajl sa kontrolama za sve definisane sesije.

4. __Kreiranje dataseta za obuku```data_to_dataset.py```:__  
Nakon definisanja _high-level_ kontrole za sve podatke koje smo hteli da iskoristimo za obučavanje, potrebno je kreirati _dataset_ za obuku, koji se treba sastojati iz odgovarajuće distribucije podataka. Takođe se podaci mešaju, te se kreiraju _batch-evi_ od 256 podatka za obučavanje u formi X^_j_={_images, high-level control_}, Y^_j_={_steering angle_}, gde je _j_ označava broj _batch-eva_. 


5. __Arhitektura modela ```functional_conditional_end_to_end_keras_model.py```:__  
Osnova arhitekture koja je korištena za model je predstavljena u _End to End Learning for Self-Driving Cars (https://arxiv.org/pdf/1604.07316.pdf)_, uz određene izmene koje se tiču još jednog ulaza koji se odnosi na _high_level_ kontrolu.


6. __Obuka modela ```train_model_batches_control.py```:__  
Treniranje modela se vrši u _batch-evima_ dobijenim iz prethodnog koraka. Ovaj pristup je izabran kako bi se mogla lako I brzo vršiti augmentacija na podacima, I kako bi se rešio problem sa memorijom prilikom otvaranja velikih fajlova.

7. __Testiranje modela ```test_model_in_CARLA.py```:__  
Prilikom testiranja modela bitno je odabrati model koji se želi testirati, grad u _Carla_ simulatoru u kom se želi testirati model. Komande za davanje _high-level_ kontrole su strelice na tastaturi, koje određuju da auto treba skrenuti levo ukoliko se pritisne leva strelica, pravo ukoliko se pritisne strelica gore, desno ukoliko se pritisne strelica desno, te kuda kog ga model vodi, ukoliko se pritisne strelica dole. 


## Modeli

Uz kod su dostupna i dva modela, jedan obučen na _raw_ podacima, I jedan na kom je primenjena augmentacija prilikom obuke. 


Oba modela su obučena kroz 10 epoha, na relativno malom setu podataka od približno 190000 frejmova podeljenom u 745 _batch-eva_. Za optimizator je korišten _Adam_ sa _learning rate-om_ od 1e-4. 

__Bitno je napomenuti da su _raw_ podaci prikupljani u sunčanom okruženju, bez oblaka i kiše, te da se mogućnosti augmentacije vrlo očituju u promenjenom okruženju sa kišom i oblacima, gde se model obučen na _raw_ podacima ne snalazi, gde se model obučen na istim ali augmentovanim podacima snalazi bolje.__

![](Pictures/aug.gif)
