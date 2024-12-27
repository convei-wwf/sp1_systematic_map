library(tidyverse)
library(here)
library(synthesisr)

sbr_fix_key <- read_csv(here('_data/1c_refs_clean/sbr_fix_lookup.csv'))

### As fed into Colandr
sbr_fix_key <- read_csv(here('_data/1c_refs_clean/sbr_fix_lookup.csv')) %>%
  select(title, key)
sample_refs <- read_refs(here('_data/1c_refs_clean/sample/sample_1000_240708.ris')) %>%
  rename(key = notes) %>% select(-year)
round1_refs <- read_refs(here('_data/4_screen_classifier_round1/ris_to_colandr_classifier_round1_sample_200.ris')) %>%
  rename(key = notes)
round2_refs <- read_refs(here('_data/5_screen_classifier_round2/ris_to_colandr_classifier_round2.ris')) %>%
  rename(key = notes) %>% mutate(year = as.integer(year))

results_all_df <- read_csv(here('_data/screened_colandr/colandr_companion_incl_2024-12-25.csv')) %>%
  bind_rows(read_csv(here('_data/screened_colandr/colandr_companion_excl_2024-12-25.csv'))) %>%
  janitor::clean_names() %>%
  mutate(date = as.Date(date_screened_t_a)) %>%
  ### drop dupe IDs from initial screening
  group_by(id) %>%
  filter(date == max(date)) %>%
  ungroup() %>%
  mutate(across(where(is.character), str_squish)) %>%
  mutate(phase = case_when(is.na(date) ~ 'benchmark',
                           date <= as.Date('2024-04-17') ~ 'early',
                           date <= as.Date('2024-04-29') ~ 'soc ben repo',
                           date <= as.Date('2024-08-02') ~ 'sample 1000',
                           date <= as.Date('2024-10-04') ~ 'classifier round 1',
                           date <= as.Date('2024-11-01') ~ 'classifier round 2',
                           date <= as.Date('2024-11-23') ~ 'classifier round 2a',
                           TRUE ~ 'classifier round 2b')) %>%
  mutate(t_a_status = case_when(is.na(t_a_status) & str_detect(tags, 'benchmark') ~ 'included',
                                is.na(t_a_status) & !str_detect(tags, 'benchmark') ~ 'excluded',
                                TRUE ~ t_a_status)) %>%
  rename(year = publication_year, 
         screening_status = t_a_status, 
         excl_reasons = t_a_exclusion_reasons)

table(results_all_df$phase)

### df of dupes TO BE DROPPED using anti_join
results_title_dupe <- results_all_df %>%
  mutate(title_lc = tolower(title)) %>%
  janitor::get_dupes(title_lc) %>%
  group_by(title_lc) %>% 
  mutate(mismatch = !all(screening_status == first(screening_status))) %>% 
  ### drop "excluded" mismatches OR drp later screening results
  filter((mismatch & screening_status == 'included') | date_screened_t_a != min(date_screened_t_a)) %>%
  ungroup()
  
  
results_df <- results_all_df %>% 
  anti_join(results_title_dupe) %>% 
  select(title, year, doi, tags, phase, screening_status, excl_reasons) %>% 
  distinct()

check_sbr <- results_df %>%
  filter(phase == 'soc ben repo') %>%
  mutate(title = str_to_title(title)) %>%
  anti_join(sbr_fix_key)
### no mismatches

check_sample <- results_df %>%
  filter(phase == 'sample 1000') %>%
  anti_join(sample_refs)
### no mismatches

check_round1 <- results_df %>%
  filter(phase == 'classifier round 1') %>%
  ### this still had a couple hundred from the sample round
  anti_join(bind_rows(round1_refs, sample_refs))
  ### clean - all refs accounted for

check_round2 <- results_df %>%
  filter(phase == 'classifier round 2') %>%
  anti_join(bind_rows(round2_refs, sample_refs), by = 'doi')
  ### mostly clean - all but one (excluded) ref accounted for -
  ### exception is due to malformed doi

check_round2a <- results_df %>%
  filter(phase == 'classifier round 2a') %>%
  ### still processing refs predicted my classifier round 2
  anti_join(bind_rows(round2_refs, sample_refs), by = 'doi')
  ### mostly clean - all but three refs accounted for -
  ### again exceptions are due to malformed doi

check_round2b <- results_df %>%
  filter(phase == 'classifier round 2b') %>%
  ### still processing refs predicted my classifier round 2
  anti_join(bind_rows(round2_refs, sample_refs), by = 'doi')
  ### mostly clean - all but three refs accounted for -
  ### again exceptions are due to malformed doi

today <- Sys.Date()
write_csv(results_df, sprintf(here('_data/screened_colandr/colandr_by_phase_%s.csv'), today))

