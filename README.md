# BreastCare AI — prototype Streamlit

## Lancer l'application
1. Décompresser le ZIP.
2. Copier `modele_vgg16.keras` dans ce dossier (ou le charger via la barre latérale).
3. Installer Python puis exécuter :
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```

## Classes du modèle
Dans la barre latérale, remplacer Classe 0/1/2 par les libellés EXACTS et dans le même ordre que `class_indices` pendant l'entraînement. Ne pas deviner l'ordre.

## Limites médicales
Prototype académique uniquement. Les scores ne sont pas des probabilités cliniques validées et Grad-CAM ne prouve pas la localisation d'une lésion. Ne pas utiliser pour diagnostiquer ou décider d'un traitement. Validation clinique, réglementaire, sécurité et protection des données indispensables avant tout usage hospitalier. Les données restent uniquement dans la session Streamlit.
