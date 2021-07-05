This contributing guideline is adapted from that of R package [hyperSpec](https://github.com/cbeleites/hyperSpec). 

:+1::tada: First off, thanks for taking the time to contribute! :tada::+1:

The following is a set of guidelines for contributing to the `Python` package `PENSA`.

## Code Licensing

License of **PENSA** project to is MIT license. By contributing, you understand and agree that your work becomes the part of the **PENSA** project and you grant permission to the **PENSA** project to  license your contribution under the [MIT license](https://opensource.org/licenses/MIT) or a compatible license.

## Reporting Bugs and Submitting Suggestions

* Ensure the bug was not already reported by searching on GitHub under [Issues](https://github.com/cbeleites/PENSA/issues).
* If you're unable to find an open issue addressing the problem, open a new one. Be sure to include a title and clear description, as much relevant information as possible, and a code sample or an executable test case demonstrating the expected behavior that is not occurring.
* Refer to the [Bug Reporting in R article](https://www.r-project.org/bugs.html) for guidance.
* The ideal minimal working example is a unit test.

## Color Use (Vignettes & Perhaps Examples)

<!-- 
* There is a built-in colorblind friendly qualitative palette `palette_colorblind`; please use whenever possible. 
* These next two palettes are also colorblind-friendly:
  - For a diverging palette, please use `colorspace::diverging_hcl(n, palette = "Blue-Red2")`.
  - For a sequential palette, please use `colorspace::sequential_hcl(n, palette = "viridis")`.
 -->


## Code and Documentation Style Guide

<!-- 
* This project adheres to the [Tidyverse styleguide](https://style.tidyverse.org/).
* This guide applies both to the code and the [`roxygen2` documentation](https://style.tidyverse.org/documentation.html).  
* The documentation is written in markdown
  - Whenever touching a function, please take the time to check its documentation.
    Large parts have been converted automatically from LaTeX, and may still need some manual polishing.

* Use package [styler](http://styler.r-lib.org/) with RStudio add-in to easily re-style your code to comply with the guidelines.
* If a unit test needs to be disabled temporarily, please use `skip("reason for switching off")`.
  This way, we'll be reminded that the test is switched off whenever the unit tests are run.
 -->

## Vignette Style and Standards
<!-- 
* No multi-line sentences (enter/return after each sentence, extra return starts a new paragraph).  This facilitates comparison via version control.
* Package names in text are written in **bold**.
* Package names in section headings are in **_bold italic_**.
* In section headings main words should be capitalized.
* Use the British English variant and corresponding spell checker.
* Function names should be written between backticks with the parentheses and followed by a signal to use syntax formatting (not simply for example `texttt` or `code` if one was using pure LaTeX), for example `fun()`{.r}.  The CSS file controls the exact appearance.
* When parts of the `PENSA` objects are mentioned, use the syntax formatting, for instance `@data`{.r}.
* When parts of a structure such as a column or list element, use the syntax highlighting, e.g. `$something`{.r}.
* Figures should have captions.
* Every figure should be mentioned in text via reference.  This will give automatic numbering.
* Figure and Table should be capitalized when mentioned in text, e.g. Figure 1.  Do not use "Fig.".
* Mention of the software `R` should be formatted with backticks.
* Sourced files should make vignette authors' life easier, and should not contain anything important to end users/readers.
* The first code block should start with `rm(list = ls())` to ensure a clean, reproducible workspace.
* If you need to leave a note in a vignette, please use this [method](https://github.com/cbeleites/PENSA/pull/147#issuecomment-646685392).
* Figure sizes are largely standardized, via the hook defined in `vignette-default-settings.R`.  See the discussion there.
 -->


## Working With Git

### Branches
This project follows the [`git flow` branching model](https://nvie.com/posts/a-successful-git-branching-model/).

<img src='https://nvie.com/img/git-model@2x.png' width='400px'>

#### Briefly

The branch `master` contains stable releases that are tested and guaranteed to work. It is not allowed to contribute directly to `master`.

The branch `develop` contains latest delivered development changes for the next release. When `develop` reaches a stable point and is ready to be released, it gets merged to `master` and tagged with a version number (e.g. 'v0.99.21'). This procedure is a subject for a strict review.

You *should not* directly contribute to `develop`, unless the change is trivial (e.g. a typo). Instead, for any new feature or bugfix, please create a separate supporting branch. We use a default naming convention for them:

* `feature/###-<feature_name>` for new features. Generally, for a new feature you should open an issue which *at least* describes the intended feature; it may go further and allow for discussion and refinement before much effort is expended.  `###` is the corresponding [issue number](https://github.com/cbeleites/PENSA/issues).
* `bugfix/###-<bugfix_name>` for bugfixes
* `release/x.y.z` for release preparation, where `x.y.z.` is the version to be released. See section "Release process" below for details.

It is recommended to use the `git flow` tool to streamline the process (see [Cheatsheet for git flow](https://danielkummer.github.io/git-flow-cheatsheet/)). However, do not call `git flow xxx finish` as it makes a merge *without* the code review - instead, [finish your branch by opening a pull request](https://softwareengineering.stackexchange.com/a/189062/302312).

Please make sure that the package can be built and and that all checks and unit tests are passed before merging back into `develop`. The shortcut in RStudio for that is `Ctrl`+`Shift`+`E`.

If you are making a significant change, please also add an entry to `NEWS.md`.

#### Wait, What if I'm not Allowed to Create a Branch in the Main Repository?

If you are not a member of the project then you cannot create a branch in the main repository. But this is not a problem! In this case, you simply fork the main repository, make the changes starting off the `develop` branch, and merge it back into the `develop` branch of the main repository via a pull request.

After a successful code review the pull request gets accepted, and your changes are represented in the main repo as a separate branch (in accordance with our guidelines). After that you can delete your fork, if you'd like.

### Pull Requests

Open a pull request via GitHub interface to let others see your work and review it. It is a collaborative tool, so we encourage you to open a ['draft pull request'](https://github.blog/2019-02-14-introducing-draft-pull-requests/) as soon as you start working on your part. This provides a place for the community to discuss your work and correct it as you go. Once your part is completed, change the status to “Ready for review”.

> The project maintainer may want to merge your pull request when your work is usable, even before it is 100% complete. In such a case, the branch must be deleted and a new one created off the `develop` branch. You can use the same name for it to indicate that this is a continuation. It will be merged, as usual, via a new pull request. This may seem to be an overhead at first glance, but it leads to a faster integration and makes the the pull requests smaller and less overwhelming.

The merged support branches [should be deleted - they're clutter](https://ardalis.com/why-delete-old-git-branches). If you want to keep their name for reference, just apply a `git tag` after the merge. Never reuse merged branches, [it can lead to problems](https://stackoverflow.com/a/29319178).


### Git Commits

Commit often, try to make small atomic commits.
An atomic commit addresses only a small separate fix or change and is more or less self-consistent.
Every commit should be related to one feature only, but the commit should group strongly related changes together (e.g. when refactoring to rename a function, all files that are affected by this should be in the same commit).

### Commit Messages

* Use the present tense ("Add feature" not "Added feature")
* Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
* Limit the first line to 72 characters or less
* Reference issues and pull requests liberally after the first line
* Give a high-level description of the what and *why* of the changes
  (similar to good code comments) already in the first line
* Use the most specialized verb that describes the situation

## Versioning

The project adheres to the semantic versioning guidelines, as outlined at https://semver.org/ (Work in progress, see *an example issues*).

Briefly, the version string has the form `x.y.z` (or `major.minor.patch`), where the major number gets incremeted if a release introduces breaking changes, the minor one after any changes in functionality (new features of bugfixes), and the patch number is increased after any trivial change. If a major or minor number is incremented, all subsequent ones are set to zero.

The version numbers refer only to commits in the `master` branch, and get incremented in one of two cases:
* during the release preparation, when a `release/x.y.z` branch buds off `develop` and merges into `master`.
* after a hotfix, which also results in a new commit on `master`.
* development branches have version `x.x.x.9000` (or `.9001` and so on - but that is rarely needed).  
  This is important since **pkgdown** uses the `.9000` to distinguish between documentation for the released version vs. the development version.

### Release Process
The process starts when the package is in a stable state that can be released to PyPI (release candidate). First, decide on a new version number `x.y.z` based on the severity of changes. Then:

* Create a `release/x.y.z` branch using `git flow release start <x.y.z>` and push it with `git flow publish`
* Open a pull request that merges into `master`
* Update the version number in the `DESCRIPTION` file
* Verify that the changes are listed in `NEWS.md`
* Confirm that the package can be built for each plaftorm
<!-- * Check for our workflow
* Ensure that all check are passed on the tarballs you build (either on your machine or using CI) with `R CMD check --as-cran <package.tar.gz>`. The checks must pass for `R` versions `R-oldrel`, `R-release`, `R-patched`, and `R-devel`. -->
* If any bugs are found, they must be fixed in the very same branch (see [here](https://stackoverflow.com/a/57507373/6029703) for details)
* Once everything works use `git flow release finish <x.y.z>`. It will merge the release branch into both `master` and `develop`, and will assign a tag to the newly created commit in the `master` branch.

<hr>

Thanks! :heart:
