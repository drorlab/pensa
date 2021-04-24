import MDAnalysis as mda
import pyemma
import numpy as np
import os
import requests


# -- Functions to download from online repositories


def download_from_gpcrmd(filename, folder):
    """
    Downloads a file from GPCRmd.
    
    Parameters
    ----------
        filename : str
            Name of the file to download. 
            Must be a file that is in GPCRmd.
        folder : str
            Target directory.
            The directory is created if it does not exist.

    """  
    print('Retrieving file',filename,'from GPCRmd.')
    url = 'https://submission.gpcrmd.org/dynadb/files/Dynamics/'
    url += filename
    req = requests.get(url, allow_redirects=True)
    out = os.path.join(folder,filename)
    os.makedirs(folder, exist_ok=True)
    open(out, 'wb').write(req.content)
    return


def get_transmem_from_uniprot(uniprot_id):
    """
    Retains transmembrane regions from Uniprot (first and last residue each).
    This function requires internet access.
    
    Parameters
    ----------
        uniprot_id : str
            The UNIPROT ID of the protein.
        
    Returns
    -------
        tm : list 
            List of all transmembrane regions, represented as tuples with first and last residue ID.
        
    """
    url = 'https://www.uniprot.org/uniprot/'+uniprot_id+'.txt'
    r = requests.get(url, allow_redirects=True)
    c = r.content
    tm = []
    for line in c.splitlines():
        if line.startswith(b'FT   TRANSMEM'):
            l = str(line)
            l = l.replace('b\'FT   TRANSMEM        ','')
            l = l.replace('\'','')
            s = l.split('.')
            tm.append((int(s[0]),int(s[-1])))
    for tmi in tm: print(*tmi)
    return tm

