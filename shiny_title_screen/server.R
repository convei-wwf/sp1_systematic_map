
# Define server logic required to draw a histogram
server <- function(input, output) {
  
  # bslib::bs_themer()

  ###################################
  ### Read and display raw bibtex ###
  ###################################
  
  v <- reactiveValues(bib_all = bib_all,
                      bib_screened = bib_screened,
                      bib_toscreen = bib_toscreen,
                      current_doc  = bib_toscreen %>% slice(1))
  
  # observeEvent(input$bib_file, {
  #   message('in bib_all reactive')
  #   roots = c(wd='.')
  #   shinyFileChoose(input = input, id = 'bib_file', roots = roots, filetypes=c('', 'txt', 'R'))
  #   f <- parseFilePaths(roots = roots, selection = input$bib_file)
  #   browser()
  #   if(is.null(f)) return(null_bib)
  # 
  #   df_out <- read_refs(f$datapath) %>%
  #     clean_bib()
  #   v$bib_all <- df_out
  # })
  
  # observeEvent(input$screened_file, {
  #   # browser()
  #   message('in bib_screened reactive')
  #   f <- input$screened_file
  #   # browser()
  #   if(is.null(f)) return(null_bib)
  #   
  #   if(f$size == 0) {
  #     df_out <- null_bib
  #   } else {
  #     df_out <- read_refs(f$datapath) %>%
  #       clean_bib()
  #   }
  #   
  #   v$bib_screened <- df_out
  # })
  
  # observeEvent(input$merge_bibs, {
  #   message('in bib_toscreen')
  #   v$bib_toscreen <- anti_join(v$bib_all, v$bib_screened)
  #   v$current_doc <- v$bib_toscreen %>% slice(1)
  # })
  
  output$toscreen_preview <- renderDataTable({
    df <- switch(input$df_preview,
                 all      = v$bib_all,
                 screened = v$bib_screened,
                 toscreen = v$bib_toscreen) %>%
      select(author, title, journal, year)
    
    DT::datatable(df)
  })
  
  
  ###################################
  ###  Screen and update output   ###
  ###################################

  observeEvent(input$skip_doc, {
    message('in doc eventReactive')
    ### update the checkbox input to blank out selections
    updateRadioButtons(inputId = 'screen_decision', selected = character(0))
    ### drop the current first row from the bib_toscreen
    v$bib_toscreen <- v$bib_toscreen %>%
      slice(-1)
    ### choose the new first row to operate upon as a new doc
    v$current_doc <- v$bib_toscreen %>%
      slice(1)
  })
  
  observeEvent(input$screen_action, {
    message('in screen_action observeEvent')
    if(length(input$screen_decision) == 0) {
      message('No decision selected! (zero length)')
      return(NULL)
    }
    if(is.null(input$screen_decision)) {
      message('No decision selected! (null)')
      return(NULL)
    }
    ### Translate current doc to RIS and add in a PA (personal note) field with the screening decision
    out_ris <- write_refs(v$current_doc, format = 'ris', file = FALSE) %>%
      paste0(collapse = '\n') %>%
      str_replace('ER  -', paste('SD  -', input$screen_decision, '\nER  -\n\n'))
    
    message(out_ris)

    ### Append the out_ris with EXTRA field to the screened output file
    write_file(out_ris, bib_screened_f, append = TRUE)
    
    ### update the checkbox input to blank out selections
    updateRadioButtons(inputId = 'screen_decision', selected = character(0))
    ### drop the current first row from the bib_toscreen
    v$bib_toscreen <- v$bib_toscreen %>%
      slice(-1)
    ### choose the new first row to operate upon as a new doc
    v$current_doc <- v$bib_toscreen %>%
      slice(1)
  })
  
  output$doc_fields_text <- renderUI({
    ### output to display selected doc for screening: highlight search terms in title and abstract
    title <- v$current_doc$title %>% str_to_sentence() %>% str_remove_all('\\{|\\}')
    title_out <- embolden(text = title) %>%
      str_replace_all('p>', 'h3>') ### turn into a header instead of paragraph
    
    author <- v$current$author %>% str_to_title() %>% markdown()
    journal <- v$current_doc$journal %>% str_to_title() %>% markdown()
    abstract <- v$current_doc$abstract %>% markdown()
    html_out <- paste(title_out, '<hr>', author, journal, '<hr>', abstract)
    # html_out <- paste(title_out, '<hr>', authors, journal)
    return(HTML(html_out))
  })

}

