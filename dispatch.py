import subprocess
from tests import get_random_states

main_file = 'tests.py'
if __name__ == '__main__':

    random_states = get_random_states()
    ntasks = len(random_states)
    print("Quick summary")
    print('Number of tasks dispatched:', ntasks)

    launcher, prototype = 'sbatch', "submit_graham.sh"

    with open(prototype) as f:
        content = f.read()
        content = content.format(ntasks=ntasks, main_file=main_file)
    with open(prototype, 'w') as f:
        f.write(content)
    subprocess.Popen([launcher, prototype])

