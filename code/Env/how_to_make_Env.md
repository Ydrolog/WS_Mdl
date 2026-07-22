# pixi Env guide
-----------------------------------------
This project uses **pixi** for managing dependencies. Why? So that Sims are reproducible and old code works.
It would be possible to use other package managers, but pixi was chosen because it is used by Deltares in imod_python, which this project is heavily based on. It's also much more convenient than conda IMO.
For more on pixi: https://pixi.sh/latest/

This guide explains how to use **pixi** to create, reproduce, and share the software environment for the *WS_Mdl* project.


## 1. Download and install pixi.
- either follow these instructions: (https://pixi.sh/latest/installation/)
	for Win, you can run the following command in PowerShell.
	powershell -ExecutionPolicy ByPass -c "irm -useb https://pixi.sh/install.ps1 | iex"
- or if you are using conda (I haven't tried this, but it should work):
	install -c conda-forge pixi


## 2. Clone env.
If you don't have the files for this project, you'll need to **clone** the repo.

You're strongly advised to create a drive (G:) for this project, as it makes paths shorter and guarantees they you won't have to make any path fixes in configuration files later on.
Relative paths used in this or other guides start from the repo root (G:) unless specified otherwise.

cd G:\
git init
git config --global --add safe.directory G:/ # tells git that G: is safe (otherwise next step fails)
git remote add origin https://github.com/Ydrolog/WS_Mdl.git
git fetch origin
git checkout -t origin/main

If you're working in another folder, you can use:
git clone <path>
But you're not allowed to clone directly on G:, as it's a root dir.


## 3. (re)create (previous) pixi env.
(This can be replaced by a script that only requires the MdlN as input to reproduce the env)
If you want to update to the latest version of the repo, skip steps 2-4 (inclussive).
The main reason to recreate an old pixi env is to reproduce an older Sim, as it ensures identical Sim conditions, thus identical results.

1. Ensure you've committed any work before doing this. (git add, git commit etc. or git stash)

2. Register the Sim you're planning to repeat in ./Mng/RunLog with a new RunN. It's safer and simpler to make new Sims than to remove/overwrite the old ones. "Sim numbers are free, the pain of Sim confusion is priceless"
Make sure you write in the description which Sims you're repeating and how you're recreating them (could be a reference to this file...)

3. **Copy** the hash (or tag) of the Sim you want to recreate.
*Open* `./Mng/RunLog.xlsx` and copy the *Tag* or *Hash* column for the Sim you want to re-run.

4. **Restore** env defining files:
git restore --source <sha_or_tag> --pathspec-from-file=G:\code\Env\pixi_env_Fis.txt

5. Change directory
cd G: # if cd G: didn't work, try pushd G:

6. **Re-build** env:
pixi install --frozen # This uses both the pixi.lock and pixi.toml files to ensure reinstallation of locked package versions.
pixi run install # This executes our .toml file task (install coupler/primod). I believe primod is not available via conda or pypi, that's why we do it that way.

alternatively, 
pixi install # Will install dependencies, but won't ensure identical package versions. This is probably faster and gives a more "modern build", but it's not as secure as pixi install --frozen

7. Optional: **WS_Mdl refresh**:
pixi run --manifest-path G:/pixi.toml --frozen --no-install pip install -e G:/ # (pip install -e G:\code (--use-pep517 --no-build-isolation) could also work) # This might be redundant, i.e. updates are reflected imedeately.
Run this whenever you want to update WS_Mdl. It's in edit mode, so any small changes (e.g. code in exiting files) are updated automatically. I use this when I make a new terminal tool and I want to add it to path. If you're adding a new function (e.g. to be accessed via WS_Mdl <function> <args>, or in scripts), you don't need to update WS_Mdl.

This will give you an environment identical to the Sim's.
To activate it run:
pixi shell
This needs to be run inside the repo folder. It can be run in downstream folders as well, pixi will look upstream when there isn't a pixi.lock file in the activate folder.


## 4. Download & install software
(only iMOD5 is essential, but the rest will make your life much easier)
You'll need to copy y:\research-ws-imod\Auxi\.irods\ to C:\Users\<user>\ and add paste a password to Pw.txt (instructions in folder)
pixi shell # (if not activated already)
python g:\code\Py\iBridges\software\Dl_installers.py
g:\code\build\install_MSI.ps1
python g:\code\Py\iBridges\software\Dl_iMOD5.py
for Double Commander, feel free to copy settings from:
g:\code\build\doublecmd_settings\
to:
C:\Users\<user>\AppData\Roaming\doublecmd\


## 5. Transfering large files - YoDa

0. To transfer large files within the scope of this project, we use YoDa [YoDa](https://geo.yoda.uu.nl), which is accessed through iBridges.

1. Use the scripts in code/Py/iBridges, which utilize the iB commands (WS_Mdl.io.ibridges) to transfer your files accross.

2. For folders that contain a large amount of not so big files, it's advised to bundle (.tar) and compress (.gz) them.
WSL/.tar.gz method:
	wsl # requires WSL installation
	tar -cf - <folder_name> | pv | gzip -1 > <compressed_name>.tar.gz

7z method:
	install 7z from G:/software/installers. The follwing command installs it with an exit code and adds it to path. Alternatively, you can use g:\code\build\install_MSI.ps1
		$p=Start-Process msiexec -Wait -PassThru -ArgumentList '/i "G:\software\installers\7z2601-x64.msi" /qn /norestart'; if($p.ExitCode -in 0,3010){$z="C:\Program Files\7-Zip"; $old=[Environment]::GetEnvironmentVariable("Path","User"); if($old -notlike "*$z*"){[Environment]::SetEnvironmentVariable("Path", "$old;$z", "User")}; $env:Path += ";$z"; "Installed. ExitCode: $($p.ExitCode). 7z added to user PATH."} else {"Install failed. ExitCode: $($p.ExitCode)"}
	7z a -t7z e:\models\NBr\Sim\NBr77.7z e:\models\NBr\Sim\NBr77_ -mx=1 -mmt=on

3. It's possible to connect YoDa as a drive (right click This PC -> Map Network Drive -> Folder: https://geo.data.uu.nl/, password: get one from https://geo.yoda.uu.nl/user/data_access and copy it to C:\Users\<User>\.irods\Pw.txt (reverse the Pw string, iB_get_Pw will reverse it back by default - small safety feature), or remember it some other way.


## 6. Updating and exporting the pixi env

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
