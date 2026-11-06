**How to run a simulation**
This guide explains how to run a simulation when making use of a VM on the SURF HPC Cloud. 

1.	Pull the most recent version of the Git repository on main
•	In the terminal, change directory  cd G: 
•	git switch main (to make sure you are in your main branch)
•	git pull
Note: some files that are needed during the PrP are to large for Git and need to be downloaded from YoDa (e.g. the CHDs)

2.	Make your own branch and check it out immediately 
•	git checkout -b (your name)-nbrxxx-(description)

3.	Activate the pixi environment
•	pixi shell 

4.	Register the run in Excel and make sure it is also saved in the CSV file (especially for Mac users)

5.	Generate the new configuarion files for this simulation
•	python code/WS_Mdl/scripts/S_from_B.py NBrxxx

6.	Open the newly generated configuration files and edit them, together with with the input relevant for this run. Make sure to use the correct simulation number and naming system throughout.

7.	Have a good look at all files that have been edited and saved 
•	git status

8.	Commit the changes
•	git add .
•	git commit -m “description of the branch”

9.	Synchronise with main before pushing (this includes edits in the repository that could have been made during the model preperation phase)
•	git fetch origin main
•	git merge origin/main 

10.	Push the branch
•	git push -u origin (name of the branch given at step 2)

11.	Open GitHub and do a pull request of the branch to main – this needs to be accepted by the admin before continuing 

12.	On the VM:
•	git fetch origin
•	git status
•	git pull
•	Add files that were created in the PrP and are to large to push to GitHub)

13.	Make sure there is enough space available on the E: drive on the VM and that the output of any potential reference run is still saved on the E: drive (when comparing them).
 
14.	Start the simulation on the VM:
•	In Double Commander, navigate to G: and press F9 to open the (configured) PowerShell terminal
•	Run: WS_Mdl RunMng freeze_env=off

