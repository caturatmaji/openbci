import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

'''
Definisi fungsi-fungsi untuk pengolahan data dalam format OpenBCI
oleh: Catur Atmaji 
(catur_atmaji@ugm.ac.id)
2021
'''

## Fungsi-fungsi yang berkaitan dengan direktori
def make_subj_files(record_dir='recordings\\',subj_codes=[],sess_codes=[]):
    '''
    membuat list berisi seluruh file dalam direktori
    input:
        record_dir : alamat direktori utama
        subj_codes : list berisi kode untuk setiap subjek
        sess_codes : list berisi kode untuk setiap sesi
    output:
        subj_files : dictionary berisi struktur folder
    '''
    subj_files = {'main_dir':record_dir}
    subj_dict = {}
    for folder_name in os.listdir(record_dir):
        subject,session,day = folder_info(folder_name)
        sess_day = session+'-'+f'{day:02d}'
        if subject not in subj_codes:
            subj_codes.append(subject)
        if session not in sess_codes:
            sess_codes.append(session)

        file_list = []
        for filename in os.listdir(record_dir+folder_name):
            file_list.append(filename)
        if bool(file_list):
            if subject not in subj_dict:
                subj_dict[subject] = {}
            if sess_day not in subj_dict[subject]:
                subj_dict[subject][sess_day] = []
            subj_dict[subject][sess_day] += file_list
        
    subj_files['subjects'] = subj_codes
    subj_files['sessions'] = sess_codes
    subj_files['subj_dir'] = subj_dict
    
    return subj_files
        
def folder_info(folder_name,prefix='OpenBCISession_',delim='-'):
    '''
    menampilkan informasi dari sebuah nama folder
    input:
        folder_name : nama folder yang akan dilihat informasinya
        prefix      : awalan dari nama folder
        delim       : jeda antar setiap informasi
    output:
        tuple berisi:
        subject     : kode subjek
        session     : kode sesi
        day         : nomor hari
    '''
    part = folder_name.replace(prefix,'')
    part = part.split(delim)
    subject = part[0]
    session = part[1]
    day = int(part[2])
    return (subject,session,day)

def print_filepath(subj_files=None,subject='',session='S1',day=0,trial=0,prefix='OpenBCISession_'):
    '''
    menampilkan alamat lengkap untuk sebuah trial
    input:
        subj_files : dictionary berisi struktur folder
        subject    : kode subjek
        session    : kode sesi
        day        : nomor hari
        trial      : nomor perulangan pada satu sesi per hari
        prefix     : awalan dari nama folder
    output
        filepath   : alamat lengkap dari file yang dilihat
    '''
    if subj_files == None:
        print("berkas subj_files harus dibuat terlebih dahulu.")
        return False
    
    filepath = subj_files['main_dir']
    filepath += prefix+subject+'-'+session+'-'+f'{day:02d}'+'\\'
    filepath += subj_files['subj_dir'][subject][session+'-'+f'{day:02d}'][trial]
    return filepath

def read_file(subj_files=None,subject='',session='S1',day=0,trial=0,filepath='',prefix='OpenBCISession_'):
    '''
    membaca file dan menyimpannya dalam format panda frame
    input:
        subj_files : dictionary berisi struktur folder
        subject    : kode subjek
        session    : kode sesi
        day        : nomor hari
        trial      : nomor perulangan pada satu sesi per hari
        prefix     : awalan dari nama folder
    output
        data_pd    : data dalam format panda frame
        
    '''
    if subj_files == None:
        print("berkas subj_files harus dibuat terlebih dahulu.")
        return False
    
    if filepath == '':
        filepath = print_filepath(subj_files,subject,session,day,trial,prefix)
    
    exg_pd = pd.read_csv(filepath,header=4,sep=',')
    
    return exg_pd

## Fungsi-fungsi yang berkaitan dengan koreksi awalan data exg
def gradient_lsm(X,Y,N=0):
    '''
    menghitung gradien dari pasangan (X,Y) sepanjang N-titik
    input:
        X : data variabel x
        Y : data variabel y
    output:
        m : gradien atau slope dari regresi linier pasangan (X,Y)
    '''
    if N==0:
        N = len(X)
    m = (N*np.dot(X,Y)-np.sum(X)*np.sum(Y))
    m /= (N*np.dot(X,X)-(np.sum(X))**2)
    return m

def correct_start(exg_pd,T=[1.75,3.25],fs=1600,N_lsm=100,threshold=1e-3,show=False):
    '''
    mencari titik mulai data exg yang memiliki indeks waktu yang benar sesuai fs
    input:
        exg_pd    : data dalam format panda frame
        T         : rentang waktu pencarian
        fs        : frekuensi pencuplikan  
        N_lsm     : panjang data untuk perhitungan gradien
        threshold : batas toleransi gradien 
        show      : opsi menampilkan grafik
    output:
        gradients : daftar perubahan gradien sepanjang waktu
        mark      : indeks sebagai tanda titik koreksi
        found     : status ditemukannya titik koreksi
    '''
    ms_pd = pd.to_datetime(exg_pd.iloc[:,-1])
    ms_pd = ms_pd.dt.microsecond
    ms_np = ms_pd.to_numpy()
    ms_np = ms_np/1000000
    N = [int(ti*fs) for ti in T]
    X = np.arange(N[0],N[1])/fs
    Y = ms_np[N[0]:N[1]]

    gradients = []
    found = False
    mark = N[0]
    for i in range(N[1]-N[0]-N_lsm):
        m = gradient_lsm(X[i:i+N_lsm],Y[i:i+N_lsm],N_lsm)
        gradients.append(m)
        if (1-m)<threshold and not found:
            print(m)
            mark += i
            found = True
    
    if show:
        plt.figure(figsize=[12,8])
        plt.plot(X,Y)
        plt.plot(X[:-N_lsm],gradients)
        plt.plot([mark/fs,mark/fs],[0,1.1])
        plt.xlim(T)
        plt.ylim([-0.1,1.1])
        plt.legend(['time index','gradient','start index found'])
        plt.grid()
        
    return gradients,mark,found

def correct_exg_pd(exg_pd,mark=0):
    '''
    mengoreksi data dengan menghilangkan awalan data sebelum indeks "mark"
    input:
        exg_pd : data dalam format panda frame
        mark   : indeks dimulainya data exg yang baru
    output:
        exg_pd_corrected : data dalam format panda frame yang terkoreksi
    '''
    exg_pd_corrected = exg_pd.iloc[mark:,:]
    return exg_pd_corrected

def pd_to_numpy(exg_pd,board='Ganglion',sel_chan=False):
    '''
    mengkonversi data dalam format panda frame ke format numpy array
    input:
        exg_pd   : data dalam format panda frame
        board    : board yang digunakan 
        sel_chan : isi manual daftar channel yang ingin dikonversi
    output:
        exg_np   : data dalam format numpy array
            baris : channel
            kolom : data
    '''
    if not sel_chan:
        if board == 'Ganglion':
            sel_chan = [1,2,3,4]
        elif board == 'Cython':
            sel_chan = [1,2,3,4,5,6,7,8]
    exg_pd_chan = exg_pd.iloc[:,sel_chan]
    exg_np = exg_pd_chan.to_numpy().transpose()
    return exg_np  

def initiate_trial_starts(subj_files):
    '''
    Membuat dictionary kosong untuk menyimpan waktu mulai untuk setiap perekaman data exg
    input:
        subj_files   : dictionary berisi struktur folder
    output:
        trial_starts : dictionary berisi waktu mulai untuk setiap rekaman exg
    '''
    trial_starts = subj_files['subj_dir']
    for subject in trial_starts.keys():
        for sess_day,files in trial_starts[subject].items():
            N_trial = len(files)
            start_times = [0 for n in range(N_trial)]
            trial_starts[subject][sess_day] = start_times
            
    return trial_starts
    
def insert_trial_start(start_time=2.0,trial_starts=None,subject='',session='S1',day=0,trial=0):
    '''
    Mengisi waktu mulai untuk setiap rekaman data exg
    input:
        start_time   : waktu mulai perekaman
        trial_starts : dictionary berisi waktu mulai untuk setiap rekaman exg
        subject      : kode subjek
        session      : kode sesi
        day          : nomor hari
        trial        : nomor perulangan pada satu sesi per hari
    output:
        trial_starts : dictionary berisi waktu mulai untuk setiap rekaman exg
    '''
    if trial_starts == None:
        print("berkas trial_time harus dibuat terlebih dahulu.")
        return False
    
    sess_day = session + '-' + f'{day:02d}'
    if subject not in trial_starts:
        print("subjek yang dipilih tidak ada.")
        return trial_starts
    elif sess_day not in trial_starts[subject]:
        print("sesi atau hari yang dipilih tidak ada.")
        return trial_starts
    
    N_trial = len(trial_starts[subject][sess_day])
    if trial >= N_trial:
        print("nomor trial yang dipilih tidak ada.")
    else:
        trial_starts[subject][sess_day][trial] = start_time
    
    return trial_starts

# Fungsi-fungsi yang terkait dengan grafik
def plot_marking_session(exg_np,sess_time=None,session='S1',time_start=0,channels=[0,1,2,3],fs=1600):
    '''
    menampilkan grafik exg dengan penanda untuk setiap pergantian gerakan
    input:
        exg_np     : data dalam format numpy array
        sess_time  : dictionary berisi gerakan beserta durasinya masing-masing
        session    : sesi yang dipilih
        time_start : waktu mulai perekaman gerakan pertama
        channels   : channels exg yang dipilih
        fs         : frekuensi pencuplikan
    '''
    if sess_time == None:
        print("dictionary sess_time harus dibuat terlebih dahulu.")
        return False
    
    sess_dur = sess_time[session]['dur']
    sess_act = sess_time[session]['act']
    
    N = exg_np.shape[1]
    tt = np.arange(N)/fs
    n_start = int(time_start*fs)
    t_start = time_start
    for ch in channels:
        exg_ch = exg_np[ch,:]
        max_exg = max(exg_ch)
        min_exg = min(exg_ch)
        plt.figure(figsize=[15,5])
        plt.plot(tt,exg_ch)
        t_now = t_start
        plt.plot([t_now,t_now],[min_exg,max_exg],'r')
        for i in range(len(sess_dur)):
            t_now += sess_dur[i]
            plt.plot([t_now,t_now],[min_exg,max_exg],'r')
    
    return True