# NEW GUIDE - env managemet py pixi ##########
-----------------------------------------
This project uses pixi for managing dependencies. Why? Cause it's more robust/secure and fast than conda or any other package manager. 
For more details check: https://pixi.sh/latest/

This guide explains how we use **Pixi** to create, reproduce, and share the software environment for the *WS_Mdl* project.

## 1. Download and install pixi.
- either follow their instructions on the website (https://pixi.sh/latest/installation/).
- or if you have/prefer conda (I haven't tried this, but it should work):
	install -c conda-forge pixi 

## 2. To re-create the env that was used for Sim.
1. If you don't have the files for this project, you'll need to **clone** the repo.
git clone https://github.com/Ydrolog/WS_Mdl C:\OD\WS_Mdl
C:\OD\WS_Mdl is the default location. If you want it somewhere else, feel free to change the path, but that may make things more complicated later.
Not all files re public, you'll need to request the rest from the project owner(s).

2. The pixi env is installed in C:\OD\WS_Mdl\code, as there are no scripts in the other folder that use it. Make it your active dir if not already true:
cd C:\OD\WS_Mdl

3. **Copy** tag or hash.
   *Open* `./Mng/RunLog.xlsx` and copy the *Tag* or *Hash* column for the run you want.

4. **Checkout** that commit:
git checkout <hash/tag>

5. **Re-build** env:
pixi install --frozen

6. **WS_Mdl** lib:
pip install -e C:\OD\WS_Mdl\code --use-pep517 --no-build-isolation

This will give you an environment identical to the Sim's.
To activate it run:
pixi shell
This needs to be run inside the repo folder. It can be run in downstream folders as well, pixi will look upstream when there isn't a pixi.lock file in the activate folder.

## 3. To export.
1. You don't have to do anything manually. If you've editted the WS_Mdl package, the new tag and hash will be recorded in the RunLog. Then the next time follow the guide in paragraph 2.
	In case you want to share with someone who doesn't use pixi, it might be possible via:
	pixi workspace export conda-environment env.yml
2. If you want to **install a new library** or switch to a specific version, use:
pixi add --pypi "rasterio>=1.3"

3. **Solve & lock**
   pixi install          # updates pixi.lock

3. **Freeze**. Don't forget to add the changes to your next commit, e.g.:
   git add pixi.toml pixi.lock
   git commit -m "Add rasterio 1.3+"
-----------------------------------------


# THE GUIDE BELOW IS THE OLD ONE. That way of env management applies from 04/08, NBr32, hash b5dfc2b backwards.
-----------------------------------------
--- To freeze/export: ---
The WS env should be frozen on every change. If no Env is present for a run/Sim you want to repeat, assume that the previous one will do the job.

1. activate:
conda activate WS

2. freeze (run this in this folder):
python .\freeze_env_WS.py

This will create a .yml file for the conda and pip dependencies
WS_Mdl is the only lib that has to be installed separatelly. 
-----------------------------------------


-----------------------------------------
--- To import/install: ---
1. install from .yml
mamba env create -f ./WS_env_<MdlN>.yml
# Can use conda instead of mamba

2. activate env
conda activate WS

3. (optional) check conda packages
conda list

4. (optional) check pip packages
pip list

5. Install WS_Mdl. You'll need to be on the WS_Mdl repo version that corresonds to your MdlN (repo_V Col in RunLog). Then run:
pip install -e C:\OD\WS_Mdl\code --use-pep517 --no-deps --no-build-isolation
-----------------------------------------