library(tidyverse)
library(here)
library(synthesisr)

sbr_fix_key <- read_csv(here('_data/3_refs_clean/sbr_fix_lookup.csv'))

### As fed into Colandr
sbr_fix_key <- read_csv(here('_data/3_refs_clean/sbr_fix_lookup.csv')) %>%
  select(title, key)
sample_refs <- read_refs(here('_data/3_refs_clean/sample/sample_1000_240708.ris')) %>%
  rename(key = notes) %>% select(-year)
round1_refs <- read_refs(here('_data/5c_classifier_round1/predicted_classifier_round1_sample_200.ris')) %>%
  rename(key = notes)
round2_refs <- read_refs(here('_data/5d_classifier_round2/predicted_classifier_round2_includes.ris')) %>%
  rename(key = notes)

results_df <- read_csv(here('_data/4_colandr_screened/colandr_companion_incl_2024-11-29.csv')) %>%
  bind_rows(read_csv(here('_data/4_colandr_screened/colandr_companion_excl_2024-11-29.csv'))) %>%
  janitor::clean_names() %>%
  mutate(date = as.Date(date_screened_t_a)) %>%
  mutate(across(where(is.character), str_squish)) %>%
  group_by(id) %>%
  filter(date == min(date)) %>%
  # ### keep only those after the date of previous download
  # filter(date_screened_t_a >= as.Date('2024-08-02')) %>%
  # ### exclude any after the Oct 4 results download
  # filter(date_screened_t_a <= as.Date('2024-10-04')) %>%
  mutate(phase = case_when(is.na(date) ~ 'benchmark',
                           date <= as.Date('2024-04-17') ~ 'early',
                           date <= as.Date('2024-04-29') ~ 'soc ben repo',
                           date <= as.Date('2024-08-02') ~ 'sample 1000',
                           date <= as.Date('2024-10-04') ~ 'classifier round 1',
                           date <= as.Date('2024-11-01') ~ 'classifier round 2',
                           TRUE ~ 'classifier round 3')) %>%
  mutate(t_a_status = case_when(is.na(t_a_status) & str_detect(tags, 'benchmark') ~ 'included',
                                is.na(t_a_status) & !str_detect(tags, 'benchmark') ~ 'excluded',
                                TRUE ~ t_a_status)) %>%
  select(title, year = publication_year, doi, tags, phase,
         screening_status = t_a_status, excl_reasons = t_a_exclusion_reasons)

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
  # anti_join(bind_rows(round2_refs, sample_refs))
  anti_join(round2_refs)
  ### mostly clean - all but one (excluded) ref accounted for

today <- Sys.Date()
write_csv(results_df, sprintf(here('_data/4_colandr_screened/colandr_by_phase_%s.csv'), today))
