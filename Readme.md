Se copieaza tot folderul si se muta.

Variabilele sunt puse local.

Fisierele html se afla in templates/.
Css, imagini statice (logo ...) si js se afla in static/
In media/ se afla imaginile incarcate de utilizator si cele procesate.

In master_djang/settings.py este configurat serverul django.
In master_djang/urls.py se afla linkurile catre paginile din site (aici se structureaza web-siteul).


Se schimba variabilele de initializare din views.py

Modelul (reteaua) se afla in pages/models.py

Modelul hdf5 se afla in models/

Comada de rulare (in folderul cu fisierul manage.py):
python manage.py runserver


Comenzi aditionale django:
	- django-admin startproject locatie/nume_proiect
	- cd nume_proiect
	- python manage.py startapp nume_aplicatie

nume_aplicatie adica un nume pentru aplicatie - blog, e-commerce ...
La mine numele aplicatiei este pages (adica pagini din site)
