## Screening Criteria

### Tier 1: Included (Definitely in scope)

* Title explicitly describes purpose as valuation of EO data on some objective or subjective metric of value.  EO data is explicitly mentioned; context is relevant to NASA Applied Sciences themes.
* Valuation is more than merely comparison.  Comparison may result in an estimate of *relative* value, but if that value metric is not explicitly stated, a document should be ranked in a different category.  
    * For example, Howe et al. (2022): *Comparing Sentinel-2 And Landsat 8 For Burn Severity Mapping In Western North America*: this document compares performance of two satellite sources, but the title does not explitly state on what value metric they are being compared (though the abstract may be more informative).
* A document tagged with this category will be added to Colandr as "included" and will not need to be included in a title-abstract screening round.

### Tier 2: Earth Observation context

* Title explicitly mentions EO data, but does not explicitly describe purpose as valuation of EO data.
* Context is relevant to NASA Applied Sciences themes.
* A document tagged with this category will be included in a title-abstract screening round.  Mention of satellite/EO data in the title suggests that this is an important consideration for the paper.

### Tier 3: Applied Science context

* Title does not explicitly mention EO data, but context is relevant to NASA Applied Sciences themes.
* A document tagged with this category will be included in a title-abstract screening round, as the mention of satellite/EO data might be included in the abstract.  However, demoting the mention of EO data out of the title indicates that the EO data might not be an important focus for the paper. 

### Tier 4: Excluded (not in scope)

* No reference to EO data, and context is not relevant to EO data.
* This might be caused by a spurious search term match, e.g., "satellite cells" in cultured meat applications.
* A document tagged with this category will excluded further screening.  These will be uploaded to Colandr as "excluded", to help train the ML model.
