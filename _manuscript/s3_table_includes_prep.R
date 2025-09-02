library(tidyverse)
x <- read_csv(here::here('_manuscript/s3_table_includes_raw.csv')) %>%
  janitor::clean_names() %>%
  mutate(reference = str_remove(reference, 'http.+'))


soc_ben_df <- x %>%
  select(key, starts_with(c('geoss', 'nasa'))) %>%
  pivot_longer(-key, names_to = 'tmp', values_to = 'sba') %>%
  filter(!is.na(sba)) %>%
  select(-tmp) %>%
  mutate(sba = str_squish(sba)) %>%
  mutate(sba = case_when(sba == 'Water' ~ 'Water Resources',
                         sba == 'Climate' ~ 'Climate & Resilience',
                         sba == 'Weather' ~ 'Climate & Resilience',
                         sba %in% c('Biodiversity', 'Ecosystems') ~ 'Ecological Conservation',
                         sba == 'Health' ~ 'Health & Air Quality',
                         sba == '(various)' ~ 'Various',
                         sba == '(other)' ~ 'Other',
                         TRUE ~ sba)) %>%
  distinct() %>%
  group_by(key) %>%
  filter(!(sba == 'Other' & any(sba != 'Other'))) %>%
  summarize(`Decision context` = paste0(sba, collapse = '; '))

val_type_df <- x %>%
  ### Start with Arias Arevalo categories to capture both monetary and non-monetary intrinsic;
  ### replace "eudaimonistic" with "relational"
  mutate(val_type = str_replace(value_type_arias_arevalo, 'eudaimonistic', 'relational')) %>%
  ### then check Himes categories; if any "relational" append "relational" to val_type
  mutate(val_type = ifelse(str_detect(value_type_himes, 'relational'),
                           paste(val_type, 'relational', sep = ';'),
                           val_type)) %>%
  select(key, val_type) %>%
  mutate(val_type = str_split(val_type, ';')) %>%
  unnest(val_type) %>%
  mutate(val_type = str_trim(val_type),
         val_type = str_to_sentence(val_type)) %>%
  distinct() %>%
  mutate(val_type = case_when(val_type == 'Instrumental' ~ 'Instrumental (monetary)',
                              val_type == 'Fundamental' ~ 'Instrumental (non-monetary)',
                              TRUE ~ val_type)) %>%
  group_by(key) %>%
  arrange(val_type) %>%
  summarize(`Value type(s)` = paste0(val_type, collapse = '; '))

deliberative_vec <- c('Participatory rural appraisal; rapid rural appraisal',
                      'Participant action research')
surveys_vec <- c('Surveys of preference assessments', 'Photo-elicitation surveys')

method_df <- x %>%
  select(key, c(method_1, method_2)) %>%
  pivot_longer(-key, names_to = 'tmp', values_to = 'method') %>%
  filter(!is.na(method)) %>%
  mutate(method = str_squish(str_to_sentence(method)),
         method = case_when(method %in% deliberative_vec ~ 'Non-monetary methods - deliberative',
                            method %in% surveys_vec ~ 'Surveys of preference assessments',
                            str_detect(method, 'Market price|Market cost') ~ 'Market price/cost methods',
                            TRUE ~ method)) %>%
  select(-tmp) %>%
  distinct() %>%
  group_by(key) %>%
  summarize(`Valuation method(s)` = paste0(method, collapse = '; '))

y <- x %>%
  select(reference, key, eo_data) %>%
  left_join(method_df, by = 'key') %>%
  left_join(soc_ben_df, by = 'key') %>%
  left_join(val_type_df, by = 'key') %>%
  select(-key, Reference = reference, `ESI source` = eo_data)

write_csv(y, here::here('_manuscript/s3_table_includes.csv'))
