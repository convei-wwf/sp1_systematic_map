
### Define UI for application
ui <- navbarPage(
  title = "Shiny Title Screening",
  # theme = bslib::bs_theme(bootswatch = 'morph'),


  # ### First panel: Welcome/introduction
  tabPanel(title = 'Welcome',
    ### Sidebar with a DT::datatable of the entire bib set?
    fluidRow(
      column(width = 3,
        h5('Choose a bibliography file:'),
        shinyFilesButton(
          id = 'files',
          label = 'choose file...',
          title = 'Select a bibliography file...',
          multiple = FALSE),

        h5('Choose or create file for screened results:'),
        # shinyFilesButton(
        #   id = 'screened_file',
        #   label = 'choose file...',
        #   title = 'Select a file for screening results...',
        #   multiple = FALSE),

        # h5('Remove already-screened results from bibliography:'),
        # actionButton(
        #   inputId = 'merge_bibs',
        #   label = 'Prep to screen', icon = icon('book')),
        # 
        # radioButtons(
        #   inputId = 'df_preview',
        #   label = 'Preview:',
        #   choices = c('all', 'screened', 'to screen' = 'toscreen'),
        #   selected = 'all'),
      ), ### end column 1

      ### Show a preview of the loaded bibtex
      column(width = 9,
        # DTOutput('toscreen_preview')
      ) ### end main panel
    )
  ), ### end Welcome tabPanel
  
  # ### Second panel: criteria
  # tabPanel(title = 'Criteria',
  #   fluidRow(
  #     column(width = 2),
  #     column(width = 8,
  #       includeMarkdown('criteria_long.md'))
  #   )
  # ), ### end tabPanel 2: screening criteria
  # 
  # ### Third panel: perform the screening
  # tabPanel(title = 'Screening',
  # 
  #   ### Sidebar with checkboxes for title screening criteria
  #   sidebarLayout(
  #     sidebarPanel(
  #       includeMarkdown('criteria_short.md'),
  # 
  #       ### Radio buttons for categorization
  #       radioButtons(
  #         inputId = 'screen_decision',
  #         label = 'Benchmark paper according to title?:',
  #         choices = c('Definitely in scope',
  #                     'Earth Observation context',
  #                     'Applied Science context',
  #                     'Not in scope'),
  #         selected = character(0)
  #         ), ### end of radio buttons
  #       ### * Append record to an output bibtex file with categorization in extra field
  #       actionButton(
  #         inputId = 'reveal_abstr',
  #         label = 'Reveal abstract?'
  #       ),
  #       actionButton(
  #         inputId = 'screen_action',
  #         label = 'Log it!'
  #       ),
  #       actionButton(
  #         inputId = 'skip_doc',
  #         label = 'Skip document!'
  #       )
  #     ), ### end sidebar panel
  # 
  #     ### Show information on a selected title
  #     mainPanel(
  #       htmlOutput('doc_fields_text')
  #     ) ### end main panel
  # 
  #   ) ### end sidebarLayout
  # ) ### end tabPanel for screening

) ### end fluidPage ui

