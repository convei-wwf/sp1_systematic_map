### update soc benefit citations - make a look up table of the old title/year/author to 
### match the new key

### old ones still show up from old data in Colandr; new in updated file

library(tidyverse)
library(here)
library(synthesisr)
old_sbr <- read_refs(here('_data/3_refs_clean', 'sbr_clean_240423_pre-update.ris')) %>%
  select(-abstract, -source_type, -journal, -keywords)
new_sbr <- read_refs(here('_data/3_refs_clean', 'sbr_clean_240423.ris')) %>%
  select(-abstract, -source_type, -journal, -keywords)

### complete matches: 239 out of 255
good_match <- inner_join(old_sbr, new_sbr) %>%
  rename(key = notes)

### doi_matches: 16 of 16 remaining
doi_match <- old_sbr %>%
  filter(!doi %in% good_match$doi) %>%
  inner_join(new_sbr, by = c('doi')) %>%
  select(author = author.x,
         title = title.x,
         year = year.x,
         doi,
         key = notes)

### new matches
new_match <- new_sbr %>%
  anti_join(good_match) %>%
  anti_join(old_sbr, by = 'doi')

all_match <- bind_rows(good_match, doi_match, new_match)
### write out results as lookup file
write_csv(all_match,
          here('_data/3_refs_clean/sbr_fix_lookup.csv'))

### still wrong: none!
wrong <- old_sbr %>%
  filter(!doi %in% c(good_match$doi, doi_match$doi))

