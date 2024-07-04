import subprocess
import json
import constants

if __name__ == '__main__':
    
    with open(f'{constants.prefix}_assignment.json', 'r') as file:
        assignment = json.load(file)

    for key in assignment:
        result = subprocess.run(f'jbsub -queue x86_24h -mem 20g python evaluation_worker.py -api_key {key}', stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
        
        