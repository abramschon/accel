# Data Challenge: Can wearable data be reliably imputed from non-wearable measurements?
## Eloise (Ellie) Ockenden, Munib Mesinovic, Abram (Bram) Schonfeldt

Self-reported measures of activity explain only 14-17% of variance in accelerometer measured activity (Pearce et al. 2020). Can more predictors improve the explanatory power of models to predict sensor data output?

As a first task, we frame this as a regression problem, using measurements taken as part of the UK Biobank as explanatory variables to predict the average euclidean norm of accelerometer data measured from a week's worth of activity. 

### Virtual environemnts
To initialise a virtual environment:
- `conda create -n accel python=3.9` (or 3.8 depending on what is on the VMs)
- `conda activate accel`
- `pip install -e .`

To start the virtual environment again:
- `conda activate accel`

### Creating symbolic links
It may be useful to create symbolic links to the shared data folder. This can be done using:
`ln -s <cdtshared_path> <convenient_path_name>`

### Adding requirements
Once you have pip installed packages, you can add them to the requirements.txt file using `pip freeze > requirements.txt`

### GitHub workflow:

Keep main up to date and create changes on a new branch.

- `git pull` 
- `git checkout -b new_branch` 
- `git add .` 
- `git commit -m "describe changes"` 
- `git push -u origin new_branch` 
- Log on to GitHub and submit pull request
- Merge changes and delete branch
