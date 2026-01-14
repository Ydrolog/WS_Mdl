# pixi Env guide
-----------------------------------------
This project uses **pixi** for managing dependencies. Why? So that Sims are reproducible and old code works.
It would be possible to use other package managers, but we chose pixi because they use it at Deltares for locking specific imod python hashes.
For more details check: https://pixi.sh/latest/

This guide explains how to use **pixi** to create, reproduce, and share the software environment for the *WS_Mdl* project.

## 1. Download and install pixi.
- either follow their instructions on the website (https://pixi.sh/latest/installation/)
	for Win, you can run the following command in PowerShell.
	powershell -ExecutionPolicy ByPass -c "irm -useb https://pixi.sh/install.ps1 | iex"
- or if you have/prefer conda (I haven't tried this, but it should work):
	install -c conda-forge pixi

## 2. Clone env.
If you don't have the files for this project, you'll need to **clone** the repo.
git clone https://github.com/Ydrolog/WS_Mdl G:
G: is the default location. If you want it somewhere else, feel free to change the path, but that may make things more complicated later.
Not all files are public, you'll need to request the rest from the project owner(s).

## 3. (re)create previous pixi env.
(This can be replaced by a script that only requires the Run to reproduce as input)
The main reason to recreate a pixi env is to reproduce an older Sim.
Most of the time, a Sim should run on most versions of the repo. But recreating the env ensures identical results, so you're advised to do so.

1. Ensure you've committed any work before doing this. (git add, git commit etc.)

2. Register new/repeat Sims in the RunLog. It's safer and simpler to have more Sims than to remove the old ones. "Sim numbers are free, the pain of Sim confusion is priceless"
Make sure you write in the descriptio which Sims you're repeating and how you're recreating them (could be a reference to this file...)

3. **Copy** tag or hash of the Sim you want to recreate.
   *Open* `./Mng/RunLog.xlsx` and copy the *Tag* or *Hash* column for the Sim you want to re-run.

4. **Restore** env defining files:
git restore --source <sha_or_tag> --pathspec-from-file=C:\WS_Mdl\code\Env\pixi_env_Fis.txt

5. Change directory
cd C:\WS_Mdl\code

6. **Re-build** env:
pixi install --frozen # This uses both the pixi.lock and pixi.toml files to ensure reinstallation of locked package versions.
pixi run install # This executes our .toml file task (install coupler/primod). I believe primod is not available via conda or pypi, that's why we do it that way.

alternatively, 
pixi install # Will install dependencies, but won't ensure identical package versions. This is probably faster and gives a more "modern build", but it's not as secure as pixi install --frozen

7. Optional: **WS_Mdl refresh**:
pixi run --manifest-path G:/pixi.toml --frozen --no-install pip install -e G:/ # (pip install -e G:\code (--use-pep517 --no-build-isolation) could also work) # This might be redundant, i.e. updates are reflected imedeately.
Run this whenever you want to update WS_Mdl. It's in edit mode, so any small changes (e.g. code in exiting files) are updated automatically. I use this when I make a new terminal tool and I want to add it to path.

This will give you an environment identical to the Sim's.
To activate it run:
pixi shell
This needs to be run inside the repo folder. It can be run in downstream folders as well, pixi will look upstream when there isn't a pixi.lock file in the activate folder.

## 4. To export.
1. You don't have to do anything manually. If you've editted the WS_Mdl package, the new tag and hash will be recorded in the RunLog. Then the next time follow the guide in paragraph 2.
	In case you want to share with someone who doesn't use pixi, it might be possible via:
	pixi workspace export conda-environment env.yml
2. If you want to **install a new library** or switch to a specific version, use e.g.:
pixi add --pypi "rasterio>=1.3"

3. **Solve & lock**
   pixi install          # updates pixi.lock

4. **Freeze**. Don't forget to add the changes to your next commit, e.g.:
   git add pixi.toml pixi.lock
   example commit:
   git commit -m "Added rasterio 1.3+"
-----------------------------------------


# THE GUIDE BELOW IS THE OLD ONE. That way of env management was used for the last time on 04/08, NBr32, hash b5dfc2b.
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
pip install -e G:\code --use-pep517 --no-deps --no-build-isolation
-----------------------------------------