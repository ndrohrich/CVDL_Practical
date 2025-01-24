import fabric
from fabric import Connection, task
from fabric import Config
from fabric import SerialGroup as Group
from fabric import ThreadingGroup as Group
from fabric import Connection
from fabric import Config
from fabric import task
import os
import sys
import tqdm
import getpass

remote_path = 'work/remote'
host = "remote.cip.ifi.lmu.de"
psw ='' #getpass.getpass(prompt='Enter LMU CIP password: ')
remote_dir = 'work/remote'
repo_url = 'https://github.com/ndrohrich/CVDL_Practical.git'
local_config_path = 'configs/config.yaml'
remote_config_path = 'work/remote/configs/config.yaml'
analysis = True

def establish_base_connection(user):
    """
    Establish a base connection to the remote host.fa

    Args:
        user (str): Username for the remote host.

    Returns:
        Connection: Fabric connection object.

    Raises:
        SystemExit: If connection fails.
    """
    
    global psw
    if psw=='':
        psw = getpass.getpass(prompt='Enter LMU CIP password: ')
    
    base_conn = Connection(host=host, user=user, connect_kwargs={"password": psw})
    try:
        base_conn.open()
    except Exception as e:
        print(f"Connect to {user}@{host} failed: {e}")
        base_conn.close()
        sys.exit(1)
    print(f"Connected to {user}@{host}")
    return base_conn

def establish_gpu_connection(user, node):
    """
    Establish a connection to a specific GPU node via the base connection.

    Args:
        user (str): Username for the remote host.
        node (str): Node to connect to.

    Returns:
        Connection: Fabric connection object.

    Raises:
        SystemExit: If connection fails.
    """
    base_conn = establish_base_connection(user)
    target_conn = Connection(host=str(node), user=user, connect_kwargs={"password": psw}, gateway=base_conn)
    try:
        target_conn.open()
    except Exception as e:
        print(f"Connect to {user}@{node} failed: {e}")
        target_conn.close()
        sys.exit(1)
    return target_conn

def analyse_gpu(conn):
    """
    Analyse GPU usage on the connected node.

    Args:
        conn (Connection): Fabric connection object.

    Returns:
        str: GPU usage statistics.
    """
    try:
        output = conn.run('nvidia-smi', hide=True).stdout
        usedwatt, sp, totalwatt = output.split('\n')[9].split()[4:7]
        return f"{usedwatt + ' ' * (4 - len(usedwatt))}/{totalwatt}"
    except Exception as e:
        print(f"Failed to analyse GPU: {e}")
        return "N/A"

@task
def connect(c, user):
    """
    Connect to a remote node and optionally analyse GPU usage.

    Args:
        c (Connection): Fabric connection object.
        user (str): Username for the remote host.
        analysis (bool): Flag to indicate if GPU analysis is required.
    """
    base_conn = establish_base_connection(user=user)
    
    # Get available nodes
    try:
        nodes = base_conn.run('sinfo', hide=True).stdout
        available_nodes = []
        for node in nodes.split('\n'):
            if 'alloc' in node and 'NvidiaAll' in node:
                available_nodes = (node.split()[-1].split(','))
                break
    except Exception as e:
        print(f"Failed to retrieve nodes: {e}")
        base_conn.close()
        sys.exit(1)
    
    if not available_nodes:
        print("No available nodes found")
        base_conn.close()
        sys.exit(1)
    
    if analysis:
        print("  |    node   | gpu_stats |")
        print("  |-----------|---------- |")
        for i, node in enumerate(available_nodes):
            print(f"{i}:| {node + ' ' * (11 - len(node) - len(str(i)))}| {analyse_gpu(Connection(host=str(node), user=user, connect_kwargs={'password': psw}, gateway=base_conn))} |")
    else:
        for i, node in enumerate(available_nodes):
            print(f"{i}: {node}")
    
    selected_node = input("Select node: ")
    selected_node = available_nodes[int(selected_node)]
    base_conn.close()
    
    target_conn = establish_gpu_connection(user, selected_node)
    watt = analyse_gpu(target_conn)
    print(f"Connected to {user}@{selected_node} with gpu_stats: {watt}")
    
    target_conn.run('bash', pty=True, hide=False)

@task
def deploy(c, user):
    """
    Deploy the repository to the remote host.

    Args:
        c (Connection): Fabric connection object.
        user (str): Username for the remote host.
    """
    base_conn = establish_base_connection(user=user)
    
    # Check the remote dir if exists if not create it
    try:
        if not base_conn.run(f'test -d {remote_dir}', warn=True).ok:
            base_conn.run(f'mkdir -p {remote_dir}')
        
        # Clone repo
        base_conn.run(f"git clone {repo_url} {remote_dir}", warn=True)
        print(f"Cloned repo to {base_conn.run('pwd', hide=True).stdout[:-1]}/{remote_dir}")
        
        # Print the repo content
        base_conn.run(f"ls {remote_dir}")
    except Exception as e:
        print(f"Failed to deploy: {e}")
    finally:
        base_conn.close()

@task
def clean(c, user):
    """
    Clean the remote directory.

    Args:
        c (Connection): Fabric connection object.
        user (str): Username for the remote host.
    """
    base_conn = establish_base_connection(user=user)
    stop = input(f"[WARNING] Do you really want to delete {base_conn.run('pwd', hide=True).stdout[:-1]}/{remote_dir}? (y/n): ")
    if stop.lower() != 'y':
        print("Aborted")
        base_conn.close()
        sys.exit(1)
    
    try:
        base_conn.run(f"rm -rf {remote_dir}")
    except Exception as e:
        print(f"Failed to clean: {e}")
    finally:
        base_conn.close()

@task
def update(c, user):
    """
    Update the remote configuration file with the local one.

    Args:
        c (Connection): Fabric connection object.
        user (str): Username for the remote host.
    """
    # Check the local config file
    if not os.path.exists(local_config_path):
        print(f"Local config file {local_config_path} not found")
        sys.exit(1)
    
    base_conn = establish_base_connection(user=user)
    
    # Check the remote dir if exists if not create it
    try:
        if not base_conn.run(f'test -d {remote_dir}', warn=True).ok:
            base_conn.run(f'mkdir -p {remote_dir}')
        
        # Check the remote config file
        if not base_conn.run(f'test -f {remote_config_path}', warn=True).ok:
            print(f"Remote config file {remote_config_path} not found")
            clone=input(f"di you want to clone the repo to {remote_dir}? (y/n): ")
            if clone.lower() == 'y':
                deploy(c, user)
            else:
                sys.exit(1)
        
        # Update the remote config file
        base_conn.put(local_config_path, remote_config_path)
        print(f"Updated {remote_config_path}")
    except Exception as e:
        print(f"Failed to update: {e}")
    finally:
        base_conn.close()

@task
def webdemo(c,model_path):
    """
    Run the web demo.

    Args:
        c (Connection): Fabric connection object.
        model_path (str): Path to the model file.
    """
    c.run(f"python3 app.py {model_path}")

@task
def help(c):
    """
    Print usage information about the available tasks.
    """
    usage_info =r"""
    Available tasks:
      connect {str:user} {bool:Analyse} - Connect to a remote node (optionally analyze GPUs)
      deploy  {str:user}                - Deploy the repository
      clean   {str:user}                - Clean the remote directory
      update  {str:user}                - Update the remote config file
      webdemo {str:model_path}          - Run the web demo for FER Classification
      help                              - Show this message
    """
    print(usage_info)


def setup_local_gitserver():
    """
    Setup a local git server.
    """
    os.system('git init --bare /tmp/gitserver')
    os.system('git clone /tmp/gitserver /tmp/gitclient')
    os.system('echo "Hello, World!" > /tmp/gitclient/hello.txt')
    os.system('cd /tmp/gitclient && git add hello.txt && git commit -m "Initial commit" && git push origin master')