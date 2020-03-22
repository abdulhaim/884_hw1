
#!/bin/bash

FOLDER_NAME="ppo_hw1"
declare -a arr=(
    "ec2-34-228-80-24.compute-1.amazonaws.com"
    )

for SSH_ADDRESS in "${arr[@]}"
do
    echo $SSH_ADDRESS

    # Pass folder that I want to train
    ssh -i ~/Documents/research/aws/marwa_key_pair.pem ubuntu@$SSH_ADDRESS "mkdir /home/ubuntu/$FOLDER_NAME"
    scp -i ~/Documents/research/aws/marwa_key_pair.pem ~/Desktop/$FOLDER_NAME/*.py ubuntu@$SSH_ADDRESS:/home/ubuntu/$FOLDER_NAME/
    scp -i ~/Documents/research/aws/marwa_key_pair.pem ~/Desktop/$FOLDER_NAME/*.md ubuntu@$SSH_ADDRESS:/home/ubuntu/$FOLDER_NAME/
    scp -i ~/Documents/research/aws/marwa_key_pair.pem ~/Desktop/$FOLDER_NAME/*.txt ubuntu@$SSH_ADDRESS:/home/ubuntu/$FOLDER_NAME/
    scp -i ~/Documents/research/aws/marwa_key_pair.pem ~/Desktop/$FOLDER_NAME/*.sh ubuntu@$SSH_ADDRESS:/home/ubuntu/$FOLDER_NAME/
    scp -i ~/Documents/research/aws/marwa_key_pair.pem ~/Desktop/$FOLDER_NAME/*.ipynb ubuntu@$SSH_ADDRESS:/home/ubuntu/$FOLDER_NAME/
    scp -i ~/Documents/research/aws/marwa_key_pair.pem ~/Desktop/$FOLDER_NAME/*.pdf ubuntu@$SSH_ADDRESS:/home/ubuntu/$FOLDER_NAME/

done
