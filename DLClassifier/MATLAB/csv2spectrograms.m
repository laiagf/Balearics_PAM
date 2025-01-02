

df = readtable("/home/laiagarrobe/Projects/Balearics_PAM/embeddings/filessw0.csv", "Delimiter", ",");
in_path = "/media/laiagarrobe/Balearic_SW/SoundTraps/Ausias March/All Recordings/selected_01/"
out_path = "/home/laiagarrobe/Projects/Balearics_PAM/embeddings/ausiasmarch_training/SW0/"
for i=1:height(df)
    disp(i)
    file = df.filesSW0{i};
    filepath = in_path+file
    getSpectrogram(filepath, 5, 48, 64, 64, out_path)
end