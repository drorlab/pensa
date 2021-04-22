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


def _extract_gpcrmd_residue_html(txt):
    res_start_line = '                        <td class="seqv seqv-sequence">'
    res_end_line = '                        </td>'
    spl = txt.split('\n')
    residue_html = []
    for lnum, line in enumerate(spl):
        if line == res_start_line:
            residue_lines = spl[lnum:lnum+12]
            # Use fewer lines if the residue is shorter
            # (i.e. has no GPCRdb number)
            if residue_lines[-4] == res_end_line:
                residue_lines = residue_lines[:-3]
            residue_html.append(residue_lines)
    return residue_html


def _extract_gpcrmd_residue_info(residue_html):
    info = []
    for res in residue_html:
        res_part = res[2].split('>')[1]
        res_seqn = res[3].split(' # ')[1][1:]
        res_code = res[-3].split(' ')[-1]
        if len(res) == 12 and 'GPCRdb' in res[5]:
            res_dbid = res[5].split(' # ')[-1][1:]
        else:
            res_dbid = ''
        info.append([res_part, res_seqn, res_code, res_dbid])
    return info


def download_gpcrmd_residues(name, directory=None):
    url = 'https://gpcrdb.org/protein/'+name+'/'
    req = requests.get(url, allow_redirects=True)
    txt = req.content.decode(req.encoding)
    residue_html = _extract_gpcrmd_residue_html(txt)
    residue_info = _extract_gpcrmd_residue_info(residue_html)
    if directory is not None:
        out_filename = os.path.join(directory,'gpcrdb-residues_'+name+'.csv')
        np.savetxt(out_filename, residue_info, delimiter=',', fmt='%s')
    return residue_info

