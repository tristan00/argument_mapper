subreddit_list = ['DebateAnAtheist',
                  'DebateAChristian',
                  'DebateAVegan',
                  'DebateReligion',
                  'PoliticalDebate',
                  'DebateACatholic',
                  'changemyview',
                  'debatemeateaters',
                  'DebateEvolution',
                  'DebateCommunism',
                  'DebateAnarchism',
                  'DebateFeminism',
                  'LeftvsRightDebate',
                  'PurplePillDebate',
                  'DebateVaccines',
                  'DebateSocialism',
                  'DebateVaccines',
                  'China_Debate',
                  'policydebate',
                  'DebateNihilisms',
                  'DebateMonarchy',
                  'CapitalismVSocialism',
                  'DebateAMuslim',
                  'debateAMR',
                  'DebateAntinatalism',
                  'DebateJudaism',
                  'Congressional_Debate',
                  'DebatePolitics',
                  'DebateTranshumanism',
                  'excatholicDebate',
                  'DebateIt']

no_user = "User deleted or not found"

main_dir = '/Users/tristandelforge/Documents/arguments'
gecko_path = '/Users/tristandelforge/Documents/arguments/geckodriver'
addon_path = '/Users/tristandelforge/Documents/arguments/reddit_enhancement_suite-5.24.6.xpi'
webpage_save_path = '/Users/tristandelforge/Documents/arguments/webpages'
json_save_path = '/Users/tristandelforge/Documents/arguments/raw_arguments'

upvote_regex = r'-?\d+'

comment_author_selector = "a.author"
view_source_class_name = "usertext-edit viewSource"
lda_max_text_length = 5000
base_dir = '/Users/tristandelforge/Documents/arguments'
lda_model_loc = f"{base_dir}/lda_model"
dictionary_loc = f"{base_dir}/dictionary"
post_link_save_loc = f"{base_dir}/posts"
topic_modeling_param_results_save_loc = f"{base_dir}/topic_modeling_param_results.csv"
llm_location = f"{base_dir}/llm"
generate_chain_aware_df_file_loc = f"{base_dir}/generate_chain_aware_df.csv"

# llm_model_id = "meta-llama/Meta-Llama-3-8B" # too big for laptop
# llm_model_id =  "meta-llama/Llama-2-7b-hf"
# llm_model_id =  "Qwen/Qwen2-7B-Instruct"
llm_model_id =  "Qwen/Qwen2-1.5B-Instruct"



