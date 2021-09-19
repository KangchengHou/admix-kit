#!/bin/bash
set -x
################################################################################
# File:    buildDocs.sh
# Purpose: Script that builds our documentation using sphinx and updates GitHub
#          Pages. This script is executed by:
#            .github/workflows/docs_pages_workflow.yml
#
# Authors: Michael Altfield <michael@michaelaltfield.net>
# Created: 2020-07-17
# Updated: 2020-07-20
# Version: 0.2
################################################################################

###################
# INSTALL DEPENDS #
###################

pip install --upgrade sphinx sphinx-rtd-theme sphinx-gallery rinohtype pygments pydata-sphinx-theme
pip install --upgrade sphinx-copybutton
#####################
# DECLARE VARIABLES #
#####################

pwd
ls -lah
export SOURCE_DATE_EPOCH=$(git log -1 --pretty=%ct)

# make a new temp dir which will be our GitHub Pages docroot
docroot=$(mktemp -d)

export REPO_NAME="${GITHUB_REPOSITORY##*/}"

##############
# BUILD DOCS #
##############

# first, cleanup any old builds' static assets
make -C docs clean

# get a list of branches, excluding 'HEAD' and 'gh-pages'
versions="$(git for-each-ref '--format=%(refname:lstrip=-1)' refs/remotes/origin/ | grep -viE '^(HEAD|gh-pages)$')"
for current_version in ${versions}; do

   # make the current language available to conf.py
   export current_version
   git checkout ${current_version}

   echo "INFO: Building sites for ${current_version}"

   # skip this branch if it doesn't have our docs dir & sphinx config
   if [ ! -e 'docs/conf.py' ]; then
      echo -e "\tINFO: Couldn't find 'docs/conf.py' (skipped)"
      continue
   fi

   ##########
   # BUILDS #
   ##########

   # HTML #
   sphinx-build -b html docs/ docs/_build/html/${current_version}

   # PDF #
   #      sphinx-build -b rinoh docs/ docs/_build/rinoh -D language="${current_language}"
   #      mkdir -p "${docroot}/${current_language}/${current_version}"
   #      cp "docs/_build/rinoh/target.pdf" "${docroot}/${current_language}/${current_version}/helloWorld-docs_${current_language}_${current_version}.pdf"

   # copy the static assets produced by the above build into our docroot
   rsync -av "docs/_build/html/" "${docroot}/"

done

# return to main branch
git checkout main

#######################
# Update GitHub Pages #
#######################

git config --global user.name "${GITHUB_ACTOR}"
git config --global user.email "${GITHUB_ACTOR}@users.noreply.github.com"

pushd "${docroot}"

# don't bother maintaining history; just generate fresh
git init
git remote add deploy "https://token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"
git checkout -b gh-pages

# add .nojekyll to the root so that github won't 404 on content added to dirs
# that start with an underscore (_), such as our "_content" dir..
touch .nojekyll

# add redirect from the docroot to our default docs language/version
cat >index.html <<EOF
<!DOCTYPE html>
<html>
   <head>
      <title>helloWorld Docs</title>
      <meta http-equiv = "refresh" content="0; url='/${REPO_NAME}/main/'" />
   </head>
   <body>
      <p>Please wait while you're redirected to our <a href="/${REPO_NAME}/main/">documentation</a>.</p>
   </body>
</html>
EOF

# Add README
cat >README.md <<EOF
# GitHub Pages Cache

Nothing to see here. The contents of this branch are essentially a cache that's not intended to be viewed on github.com.


If you're looking to update our documentation, check the relevant development branch's 'docs/' dir.

For more information on how this documentation is built using Sphinx, Read the Docs, and GitHub Actions/Pages, see:

 * https://tech.michaelaltfield.net/2020/07/18/sphinx-rtd-github-pages-1
EOF

# copy the resulting html pages built from sphinx above to our new git repo
git add .

# commit all the new files
msg="Updating Docs for commit ${GITHUB_SHA} made on $(date -d"@${SOURCE_DATE_EPOCH}" --iso-8601=seconds) from ${GITHUB_REF} by ${GITHUB_ACTOR}"
git commit -am "${msg}"

# overwrite the contents of the gh-pages branch on our github.com repo
git push deploy gh-pages --force

popd # return to main repo sandbox root

# exit cleanly
exit 0
