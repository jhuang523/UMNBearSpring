# UMNBearSpring
Repo for modeling work for Bear Spring project. 

## Repo structure
- src: helper modules for processing data, creating and running models, and processing outputs
- data: input + calibration data for model. NOTE: does not include DEM data which is too large to be hosted on git. 

- EPM: equivalent porous media models
    - steady state
    - transient
- openkarst: conduit network model using openkarst and pykasso
- visualization: notebooks and functionality for maps, figures, etc.

## Making changes
- Check out a user branch (i.e. "Charlie" or "Jenny") and start pull request when making changes. Merge with main after discussing changes.
- main will contain the most up-to-date version of the modeling work. 

## Authenticating 
- $ ssh-keygen -t ed25519 -C "your_email@example.com"
- use default location, no passphrase (just press enter through settings)
- $ vim ~/.ssh/id_ed25519.pub 
    - copy the contents
    - paste contents in github.com -> settings -> SSH keys -> add new SSH key
- $ ssh -T git@github.com
    - you should see "Hi [user]! You've successfully authenticated, but GitHub does not provide shell access."
    
