



''' ---------------------------  Channel Data ---------------------------'''
'''
    ChannelData class takes 3-args of:
        - datapath: string, pointing at '.db' file location 
        - target_table: string,
        - channel_Unique_id: string, specific channel ID
    Calling corpus() method returns:
        - the target_channel strip_text data
        - the original message content
'''


class ChannellData:

    target_channel_strip_text = {}
    original_message_content = {}

    def __init__(self,
                 datapath: 'Path to database directory',
                 target_table: 'Target Table',
                 channelUniqueId):
        self.datapath = datapath
        self.target_table = target_table
        self.channelUniqueId = channelUniqueId

    def corpus(self):

        ping.info('Saving recipes to {}'.format(self.datapath))
        db = create_engine('sqlite:///{}'.format(self.datapath))

        target_data_table = pd.read_sql(self.target_table, con=db)
        target_data_table = target_data_table.loc[target_data_table.semantic_unit_count != ""]
        target_data_table["semantic_unit_count"] = target_data_table["semantic_unit_count"].astype(
            int)

        target_ch_table = target_data_table.loc[target_data_table.ChannelUniqueID ==
                                                self.channelUniqueId]

        text_pool = target_ch_table.strip_text.loc[target_ch_table.semantic_unit_count >= 4].tolist(
        )
        self.__class__.target_channel_strip_text = text_pool

        orig_text_pool = target_ch_table.MessageContent.loc[target_ch_table.semantic_unit_count >= 4].tolist(
        )
        self.__class__.original_message_content = orig_text_pool

        return self.__class__.target_channel_strip_text, self.__class__.original_message_content



        