:+1::tada: First off, thanks for taking the time to contribute! :tada::+1:

The following is a set of guidelines for contributing to the `Python` package `PENSA`, adapted from that of the R package [hyperSpec](https://github.com/cbeleites/hyperSpec). 

## Code Licensing

The **PENSA** project is [licensed under the MIT license](https://github.com/drorlab/pensa/blob/master/LICENSE). By contributing, you understand and agree that your work becomes a part of the **PENSA** project and you grant permission to the **PENSA** project to  license your contribution under the [MIT license](https://opensource.org/licenses/MIT) or a compatible license.

## Reporting Bugs and Submitting Suggestions

* Ensure the bug was not already reported by searching on GitHub under [Issues](https://github.com/cbeleites/PENSA/issues).
* If you're unable to find an open issue addressing the problem, open a new one. Be sure to include a title and clear description, as much relevant information as possible, and a code sample or an executable test case demonstrating the expected behavior that is not occurring.
* The ideal minimal working example is a unit test.

## Code and Documentation Style Guide

* This project loosely adheres to the [PEP 8 style guide](https://www.python.org/dev/peps/pep-0008/) but this is not enforced. 
* Documentation strings are written in [NumPy style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html#example-numpy). Whenever touching a function, please take the time to check its documentation.
* Figures should be generated using `matplotlib.pyplot`. We have not settled on a more detailed style for these yet.

<!-- 
## Figures Style Guide

* Plots and figures are generated using matplotlib.pyplot
* There is no guideline for these yet
 -->
 
 
## Working With Git

### Branches
This project follows the [`git flow` branching model](https://nvie.com/posts/a-successful-git-branching-model/), starting with version 0.2.3.

<img src='https://nvie.com/img/git-model@2x.png' width='400px'>

#### Briefly

The branch `master` contains stable releases that are tested and guaranteed to work. It is not allowed to contribute directly to `master`.

The branch `develop` contains latest delivered development changes for the next release. When `develop` reaches a stable point and is ready to be released, it gets merged to `master` and tagged with a version number (e.g. 'v0.99.21'). 

You *should not* directly contribute to `develop`, unless the change is trivial (e.g. a typo). Instead, for any new feature or bugfix, please create a separate supporting branch. We use a default naming convention for them:

* `feature/###-<feature_name>` for new features. Generally, for a new feature you should open an issue which *at least* describes the intended feature; it may go further and allow for discussion and refinement before much effort is expended.  `###` is the corresponding [issue number](https://github.com/drorlab/PENSA/issues).
* `bugfix/###-<bugfix_name>` for bugfixes
* `release/x.y.z` for release preparation, where `x.y.z.` is the version to be released. See section "Release process" below for details.

Please make sure that all checks and unit tests are passed before merging back into `develop`. 
<!--
If you are making a significant change, please also add an entry to `NEWS.md`.
-->

#### Wait, What if I'm not Allowed to Create a Branch in the Main Repository?

If you are not a member of the project then you cannot create a branch in the main repository. But this is not a problem! In this case, you simply fork the main repository, make the changes starting off the `develop` branch, and merge it back into the `develop` branch of the main repository via a pull request.

After a successful code review the pull request gets accepted, and your changes are represented in the main repo as a separate branch (in accordance with our guidelines). After that you can delete your fork, if you'd like.

### Pull Requests

Open a pull request via GitHub interface to let others see your work and review it. It is a collaborative tool, so we encourage you to open a ['draft pull request'](https://github.blog/2019-02-14-introducing-draft-pull-requests/) as soon as you start working on your part. This provides a place for the community to discuss your work and correct it as you go. Once your part is completed, change the status to “Ready for review”.
<!--
The project maintainer may want to merge your pull request when your work is usable, even before it is 100% complete. In such a case, the branch must be deleted and a new one created off the `develop` branch. You can use the same name for it to indicate that this is a continuation. It will be merged, as usual, via a new pull request. This may seem to be an overhead at first glance, but it leads to a faster integration and makes the the pull requests smaller and less overwhelming.
-->
Merged support branches [should be deleted - they're clutter](https://ardalis.com/why-delete-old-git-branches). If you want to keep their name for reference, just apply a `git tag` after the merge. Never reuse merged branches, [it can lead to problems](https://stackoverflow.com/a/29319178).


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

The project adheres to the semantic versioning guidelines, as outlined at https://semver.org/. Deviations might still occur for versions `0.x.x`.

Briefly, the version string has the form `x.y.z` (or `major.minor.patch`), where the major number gets incremeted if a release introduces breaking changes, the minor one after any changes in functionality (new features of bugfixes), and the patch number is increased after any trivial change. If a major or minor number is incremented, all subsequent ones are set to zero.

The version numbers refer only to commits in the `master` branch, and get incremented in one of two cases:
* during the release preparation, when a `release/x.y.z` branch buds off `develop` and merges into `master`.
* after a hotfix, which also results in a new commit on `master`.

### Release Process
The process starts when the package is in a stable state that can be released to PyPI (release candidate). First, decide on a new version number `x.y.z` based on the severity of changes. Then:

* Create a `release/x.y.z` branch.
* Open a pull request that merges into `master`.
* Update the version number in the `setup.py` file.
* Confirm that all check are passed.
* If any bugs are found, they must be fixed in the very same branch (see [here](https://stackoverflow.com/a/57507373/6029703) for details)
* Once everything works, merge the release branch into both `master` and `develop`, and assign a tag to the newly created commit in the `master` branch.

<hr>

Thanks! :heart:
