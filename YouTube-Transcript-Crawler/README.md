''' ---------- UTubeTrnscrpt ----------- '''
'''
UTubeTrnscrpt class takes:
    - channel_name 
    - list of video  Ids 
    - desired "start" and "end" points timestamps

Calling the method "transcript" will return a dataframe 
with video Meta including the transcripts. 

'''


''' List of 150 video IDs from Computing Forever Channel'''
'''
    cfvidlist = ['x1_QEO-Xxj8','pKkWFCy2Lr0','jjvOtLdJYk0','eM3r-SW0eQs','FY7Okr0J5iE',
                '9ONa9JSMIs8','Bl9eaQyzC2Y','RqUTt7MF7HY','U-g5NxpVOVw','GLlVCQYl0eU',
                'k9XDoZ3I_yk','56886DZC0U4','jkN_RNYgwSU','LkFBtx4Cbmw','8yh3wal9mBg',
                'e3XNs4famjM','qn7hsCQ9ZXY','DlDRmce6qaw','PwE93HFUgkU','FY7KeX8Io7o',
                'bz-WSAG1tXA','y0oB67a1SVE','uNANI4-41VU','n6Thqj0hC2c','ybQU_3iZEPs',
                'uikjncdtyWA','SGq_-wnUKOM','7Pubmfh64CY','yG7asmgMYmI','pSZ36oBdIPY',
                '-t8xdoQEu2Y','FrMLZDwuFTI','xPa9zPB56_c','KvXqSF_WS2c','zIlH_nA4xTE',
                '9BwCSzfwDdg','tp1M4lP2ceQ','KahqSWHl2Jo','5BipEWSuQkg','lyRMoTaIFes',
                'Ru54nrxbm2Y','zTl9dP4VC6g','pQYse7xwq2A','E-zhiuwwCxk','u9rJBeGQFbg',
                'kYfZZ76f--E','s0iiv616ljs','tt0Xj41F5QQ','7U50294bI80','cECDuUNq6wY',
                'dMYYPztkmks','iPvHIwBM-Lk','x-wsHfLdXAM','rRGW9JZ34SE','DmBbqh4cMwc',
                'XGBRO2cAAp4','wF7dMtiI3CI','rhheWDu8ij8','070Ml5g8t44','nXHY6Fgrzck',
                'zt1vdPEOr1M','AL8XMUjCduA','_LHa8utffW4','zgONwlt8QVQ','ii_9wbk8YF8',
                'mtakByAQVAk','6UphmBMZQOU','UVi4CYRL8m8','x5smhjshMuc','RwhdXeriYzY',
                '4hgJEkzpX2U','7k_AVkOILiA','DZ9ztwaEzHs','8Li-KN8zzow','KJuoPWJj0h0',
                'W6JtidnwPK0','5ZUixDiO4Zg','Sa5TcCk0SEI','Aa868_RPdwE','No0ZBYo8ZrE',
                'UjGpHbf1m1Q','IQFxHS4PLmU','HJmFJPyW7QQ','rhU7F1pgePU','YIkVD71oVhg',
                'DFecsGd_j40','8_2xRFQyF1c','Nu8AdwNs79c','QaXdDGbDGls','Cj4LU-ZDCBo',
                'R50wVFwf5zI','6ogp_8Shmg0','vNQmCvzyrl8','6-SCswUipG0','zM_cYOeazH0',
                'fCE3Tx_Z9_w','HiMaSrXaFz0','ueIPwThAzx8','8QqrlXsA7JA','5pvhv-6Qt7o',
                'LeFNFKqLb0A','GFXYLKcAKUI','dxzFZM8KPcY','lSSW_XeR0ug','mhBt_WVPA2k',
                'QaUb1BA9M-o','LA57EvgvVA4','WZRjGy5F2EI','Fh7utAcBM54','Qrt16SaoODc',
                'jOy8Xf0FaZI','yZV4pHbcaEo','AECU058CcEE','FPSh-V4tedI','EcnZM1UWW9Y',
                'xEuEm7BCrQE','NQIucSgk4EI','b5-q7xYdTX0','EjY0FcM0ccg','qzBJZGSbXSk',
                'AxOdJtOAPe0','0RKj7Ef25N0','7e0J7kzSt1I','Pjt-vxEMpVs','4G8uubO5nHA',
                'gzrpElMAGOA','Qli6ZzLTd1k','mBXB6Ng5Hng','Lc2C7wKVZus','kgrK295oAG0',
                'yPIlBfyqwuY','CiqrZwFHkw4','eZ13ibdl3vg','lL7EpBhITNY','jp0ZLqIxITI',
                '1vU0Sxx2dCQ','zRY1vnXgx9Y','zUoumgTWHO0','-eHO5Nz2o3E','KbuzhdndZOM',
                'QmtpdTZGV04','bgaSooMfd94','75iH1b4gwnM','fitDy6oscUs','q7c551s_Cd4',
                'P1W-Pe2_Ozg','iYtyQAXkUkc','xkJnz3JRmWY','DhaZPwoS61o','NWstx2AcH9Q']

'''