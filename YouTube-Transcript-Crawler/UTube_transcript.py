from youtube_transcript_api import YouTubeTranscriptApi
import pandas as pd
import asyncio
from requests_html import AsyncHTMLSession
from bs4 import BeautifulSoup as bs
session = AsyncHTMLSession()

''' ---------- UTubeTrnscrpt ----------- '''
'''
UTubeTrnscrpt class takes:
    - channel_name 
    - list of video  Ids 
    - desired "start" and "end" points timestamps

Calling the method "transcript" will return a dataframe 
with video Meta including the transcripts. 
'''
class UTubeTrnscrpt:
    corp_documents = []
    
    def __init__(self, 
                channel_name,
                video_id_list, 
                start = None : 'Start time for transcripts in seconds', 
                end   = None : 'end time for transcripts in seconds')
        self.channel_name  = channel_name
        self.video_id_list = video_id_list
        self.start = start
        self.end   = end   
        
    
    async def transcript(self):
        data = []
        
        for vid in self.video_id_list:
        
            # Collecting the transcript
            transcript = YouTubeTranscriptApi.get_transcript(vid)
            
            if self.start == None and self.end == None:
                corp = ''
                for i in range(len(transcript)):
                    corp = corp + ' ' + list(transcript[i].values())[0]
            
            elif self.start != None and self.end != None:
                corp = ''
                for i in range(len(transcript)):
                    stpoint = list(transcript[i].values())[1]
                    if self.start <= stpoint and self.end >= stpoint:
                        corp = corp + ' ' + list(transcript[i].values())[0]     
                    
            elif self.start != None and self.end == None:
                corp = ''
                for i in range(len(transcript)):
                    stpoint = list(transcript[i].values())[1]
                    if self.start <= stpoint:
                        corp = corp + ' ' + list(transcript[i].values())[0]
                
            elif self.start == None and self.end != None:
                corp = ''
                for i in range(len(transcript)):
                    stpoint = list(transcript[i].values())[1]
                    if self.end >= stpoint:
                        corp = corp + ' ' + list(transcript[i].values())[0]  
            
            # Acquiring video Meta
            asession = AsyncHTMLSession()
            url = f"http://www.youtube.com/watch?v={vid}"
            r = await asession.get(url)
            await r.html.arender(sleep = 10)
            html = r.html.raw_html
            soup = bs(html, "html.parser")
            title        = soup.find("meta", itemprop = "name")['content']
            nviews       = soup.find("meta", itemprop = "interactionCount")['content']
            vdescription = soup.find("meta", itemprop = "description")['content']
            pubdate      = soup.find("meta", itemprop = "datePublished")['content']
            duration     = soup.find("span", {"class": "ytp-time-duration"}).text
            tags         = ', '.join([ meta.attrs.get("content") for meta in soup.find_all("meta", {"property": "og:video:tag"}) ])
            await asession.close()
            
            data_row = [self.channel_name,
                        vid,
                        transcript,
                        title,
                        vdescription,
                        nviews,
                        pubdate,
                        tags,
                        duration,
                        corp]
            
            data.append(data_row)
            
        self.__class__.corp_documents = pd.DataFrame(data, columns = ['Channel_Name',
                                                                      'Video_ID',
                                                                      'Transcript_with_timeStamp',
                                                                      'Video_Title',
                                                                      'Video_Description',
                                                                      'N_Views',
                                                                      'Date_Published',
                                                                      'Tags',
                                                                      'Duration',
                                                                      'Documnet'])
        
        # save in a csv file 
        if self.end != None :
            filename = f'{self.channel_name}_YoutubeTranscripts_(from{self.start}s_to_{self.end}s).txt'
        else:
            filename = f'{self.channel_name}_YoutubeTranscripts_fullvideo.txt'
        self.__class__.corp_documents.to_csv(filename, sep = ',', index = False, encoding = 'utf-8')
        return self.__class__.corp_documents


async def main():
    cfvidlist = ['x1_QEO-Xxj8','pKkWFCy2Lr0']
    utobj = UTubeTrnscrpt('Computing_Forever',cfvidlist,[],[])
    await utobj.transcript()
asyncio.run(main())




''' -------------------------APENDIX -----------------------------'''
''' list of 150 video_Ids from Computing Forever Youtube Channel'''
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



