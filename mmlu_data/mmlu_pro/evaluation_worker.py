import subprocess
import json
import sys
import fire
import constants
import time

def main(api_key):

    with open(f'{constants.prefix}_assignment.json', 'r') as file:
        assignment = json.load(file)
    
    print(api_key)
    print(f"Model: {constants.model_path}")

    all_commands = assignment[api_key]
    
    start = None
    speeds = []
    i = 0
    while i < len(all_commands):
        
        print("CURR INDEX: " + str(i) + "/" + str(len(all_commands)))
        sys.stdout.flush()

        command = all_commands[i]

        if start is None:
            start = time.time()
        result = subprocess.run(f"export GENAI_KEY={api_key} && " + command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)

        print("Output:")
        print(result.stdout)
        print("Error:")
        print(result.stderr)
        if result.returncode != 0:
            print(f"Command execution failed with error: {result.stderr}")
            print(f"Current iteration has been running for {time.time()-start}")
            print(f"Model: {constants.model_path}")
            if len(speeds) > 0:
                print(f'Avg Elapsed per Iteration: {sum(speeds)/len(speeds)}')
                print(f'Total Elapsed: {sum(speeds)}')
            continue
        i += 1
        end = time.time()

        speeds.append(end-start)
        
        print(f'Curr Iteration Elapsed: {end-start}')
        print(f'Avg Elapsed per Iteration: {sum(speeds)/len(speeds)}')
        print(f'Total Elapsed: {sum(speeds)}')
        print(f"Model: {constants.model_path}")
        
        start = None

    print(f'Length of all commands: {len(all_commands)}')

if __name__ == '__main__':
    fire.Fire(main)
    

        