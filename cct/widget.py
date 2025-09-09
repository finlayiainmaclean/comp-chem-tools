import anywidget
import traitlets
from rdkit import Chem
from rdkit.Chem import AllChem


class MoleculeViewer(anywidget.AnyWidget):
    _esm = """
    function render({ model, el }) {
        // Create unique IDs for this widget instance
        const widgetId = 'widget_' + Math.random().toString(36).substr(2, 9);
        const viewerId = widgetId + '_viewer';
        const labelId = widgetId + '_label';
        const contentId = widgetId + '_content';
        const molTabId = widgetId + '_mol';
        const atomTabId = widgetId + '_atom';
        const bondTabId = widgetId + '_bond';
        
        // Clear any existing content
        el.innerHTML = `<div style="display: flex; gap: 10px;"><div style="position: relative; border: 2px solid #4CAF50; border-radius: 12px; background: white; padding: 5px;"><div id="${labelId}" style="position: absolute; top: 10px; left: 10px; background: rgba(0,0,0,0.7); color: white; padding: 2px 6px; border-radius: 3px; font-size: 18px; z-index: 1000; display: none;"></div><div id="${viewerId}" style="width: 600px; height: 450px;"></div></div><div style="width: 300px; height: 450px; border: 2px solid #4CAF50; border-radius: 12px; background: white; overflow: hidden;"><div style="display: flex; border: 2px solid #4CAF50; border-radius: 8px; margin: 8px; overflow: hidden;"><button id="${molTabId}" style="flex: 1; padding: 8px 12px; border: none; background: #e8f5e8; color: #2e7d32; cursor: pointer; font-size: 18px; border-radius: 6px 0 0 0;">Molecule</button><button id="${atomTabId}" style="flex: 1; padding: 8px 12px; border: none; background: white; color: black; cursor: pointer; font-size: 18px; border-radius: 0;">Atoms</button><button id="${bondTabId}" style="flex: 1; padding: 8px 12px; border: none; background: white; color: black; cursor: pointer; font-size: 18px; border-radius: 0 6px 0 0;">Bonds</button></div><div id="${contentId}" style="padding: 10px; height: 410px; overflow-y: auto; font-family: monospace; font-size: 18px;">Loading...</div></div></div>`;
        
        const atomLabel = el.querySelector('#' + labelId);
        const propertiesContent = el.querySelector('#' + contentId);
        const molTab = el.querySelector('#' + molTabId);
        const atomTab = el.querySelector('#' + atomTabId);
        const bondTab = el.querySelector('#' + bondTabId);
        
        // Widget instance state
        let currentTab = 'molecule';
        let currentAtomProperty = null;
        let currentBondProperty = null;
        let cachedMolProps = '';
        let validAtomProperties = {};
        let invalidAtomProperties = {};
        let validBondProperties = {};
        let invalidBondProperties = {};
        let atomElements = [];
        let bondData = [];
        let viewer = null;
        let selectedAtomIndex = null;
        let selectedBondIndex = null;
        let showPropertySpheres = true;
        let showPropertyCylinders = true;
        let bondSortState = 'none'; // 'none', 'asc', 'desc'
        let atomSortState = 'none'; // 'none', 'asc', 'desc'
        
        // Ensure 3Dmol is loaded
        function ensure3Dmol() {
            return new Promise((resolve, reject) => {
                if (window.$3Dmol) {
                    resolve();
                    return;
                }
                
                if (document.querySelector('script[src*="3Dmol"]')) {
                    // Script already loading, wait for it
                    const checkInterval = setInterval(() => {
                        if (window.$3Dmol) {
                            clearInterval(checkInterval);
                            resolve();
                        }
                    }, 100);
                    return;
                }
                
                const script = document.createElement('script');
                script.src = 'https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js';
                script.onload = resolve;
                script.onerror = reject;
                document.head.appendChild(script);
            });
        }
        
        // Helper functions
        function extractAtomElements(sdfData) {
            if (!sdfData) return [];
            const lines = sdfData.split('\\n');
            const elements = [];
            
            if (lines.length > 3) {
                const countsLine = lines[3].trim();
                const atomCount = parseInt(countsLine.substring(0, 3));
                
                for (let i = 4; i < 4 + atomCount && i < lines.length; i++) {
                    const atomLine = lines[i];
                    if (atomLine.length >= 34) {
                        const element = atomLine.substring(31, 34).trim();
                        elements.push(element);
                    }
                }
            }
            return elements;
        }
        
        function extractBondData(sdfData) {
            if (!sdfData) return [];
            const lines = sdfData.split('\\n');
            const bonds = [];
            
            if (lines.length > 3) {
                const countsLine = lines[3].trim();
                const atomCount = parseInt(countsLine.substring(0, 3));
                const bondCount = parseInt(countsLine.substring(3, 6));
                
                const bondStartLine = 4 + atomCount;
                for (let i = bondStartLine; i < bondStartLine + bondCount && i < lines.length; i++) {
                    const bondLine = lines[i];
                    if (bondLine.length >= 9) {
                        const atom1 = parseInt(bondLine.substring(0, 3).trim()) - 1; // Convert to 0-based
                        const atom2 = parseInt(bondLine.substring(3, 6).trim()) - 1; // Convert to 0-based
                        const bondType = parseInt(bondLine.substring(6, 9).trim());
                        
                        const atom1Name = getAtomNameByIndex(atom1);
                        const atom2Name = getAtomNameByIndex(atom2);
                        
                        bonds.push({
                            atom1: atom1,
                            atom2: atom2,
                            bondType: bondType,
                            name: `${atom1Name}-${atom2Name}`
                        });
                    }
                }
            }
            return bonds;
        }
        
        function toTitleCase(str) {
            return str.replace(/_/g, ' ').split(' ').map(word => 
                word.charAt(0).toUpperCase() + word.slice(1).toLowerCase()
            ).join(' ');
        }
        
        function generateAtomNames(elements, numAtoms) {
            const atomNames = [];
            
            for (let i = 0; i < numAtoms && i < elements.length; i++) {
                const element = elements[i];
                atomNames.push(element + (i + 1));
            }
            return atomNames;
        }
        
        function getAtomNameByIndex(atomIndex) {
            if (atomElements && atomIndex >= 0 && atomIndex < atomElements.length) {
                const element = atomElements[atomIndex];
                return element + (atomIndex + 1);
            }
            return `A${atomIndex + 1}`;
        }
        
        function parsePropertyList(value) {
            // Support both comma-separated values and bracket notation
            // e.g., "-0.1,2,1." or "[-0.1,2,1.]"
            let cleanValue = value;
            if (typeof value === 'string' && value.trim().startsWith('[') && value.trim().endsWith(']')) {
                // Remove brackets from bracket notation
                cleanValue = value.trim().slice(1, -1);
            }
            return cleanValue.split(',').map(v => v.trim());
        }
        
        function createAtomSubTabs() {
            const atomPropKeys = Object.keys(validAtomProperties);
            if (atomPropKeys.length === 0) {
                return '<div style="padding: 20px; text-align: center; color: #666;">No valid atom properties available</div>';
            }
            
            let subTabsHtml = '<div style="display: flex; flex-wrap: wrap; border: 2px solid #4CAF50; border-radius: 8px; margin-bottom: 10px; overflow: hidden;">';
            atomPropKeys.forEach((key, index) => {
                const isFirst = index === 0;
                const isLast = index === atomPropKeys.length - 1;
                const isSelected = currentAtomProperty === key;
                
                let borderRadius = '';
                if (isFirst && isLast) borderRadius = '6px';
                else if (isFirst) borderRadius = '6px 0 0 6px';
                else if (isLast) borderRadius = '0 6px 6px 0';
                
                const bgColor = isSelected ? '#e8f5e8' : 'white';
                const textColor = isSelected ? '#2e7d32' : 'black';
                
                subTabsHtml += `<button class="atom-prop-tab" data-property="${key}" style="flex: 1; min-width: 80px; padding: 6px 8px; border: none; background: ${bgColor}; color: ${textColor}; cursor: pointer; font-size: 16.5px; border-radius: ${borderRadius};">${toTitleCase(key)}</button>`;
            });
            subTabsHtml += '</div>';
            
            const selectedProp = currentAtomProperty || atomPropKeys[0];
            currentAtomProperty = selectedProp;
            
            const values = parsePropertyList(validAtomProperties[selectedProp]);
            const atomNames = generateAtomNames(atomElements, values.length);
            
            let tableHtml = '<table style="width: 100%; border-collapse: collapse; font-size: 16.5px;">';
            tableHtml += '<thead><tr><th style="border-bottom: 1px solid #ddd; padding: 8px; text-align: left; color: #2e7d32;">Atom</th><th id="atom-value-header" style="border-bottom: 1px solid #ddd; padding: 8px; text-align: right; color: #2e7d32; cursor: pointer; user-select: none;">Value <span id="atom-sort-arrow" style="font-size: 12px;">↕</span></th></tr></thead>';
            tableHtml += '<tbody id="atom-table-body">';
            
            // Create array of valid atom data for sorting
            const atomTableData = [];
            values.forEach((value, index) => {
                // Skip NaN values
                if (value.toLowerCase() === 'nan' || value.toLowerCase() === 'n/a' || value.toLowerCase() === 'none' || value.trim() === '') {
                    return;
                }
                
                const atomName = atomNames[index] || `A${index + 1}`;
                atomTableData.push({
                    index: index,
                    name: atomName,
                    value: value,
                    numericValue: parseFloat(value)
                });
            });
            
            // Generate table rows
            atomTableData.forEach(atom => {
                const isSelected = selectedAtomIndex === atom.index;
                const bgColor = isSelected ? '#e0e0e0' : 'transparent';
                tableHtml += `<tr class="atom-row" data-atom-index="${atom.index}" data-numeric-value="${atom.numericValue}" style="background-color: ${bgColor}; cursor: pointer;"><td style="border-bottom: 1px solid #eee; padding: 6px; font-weight: bold;">${atom.name}</td><td style="border-bottom: 1px solid #eee; padding: 6px; text-align: right;">${atom.value}</td></tr>`;
            });
            
            tableHtml += '</tbody></table>';
            
            // Create toggle button at bottom right
            const toggleText = showPropertySpheres ? 'Hide Spheres' : 'Show Spheres';
            const toggleColor = showPropertySpheres ? '#2e7d32' : '#666';
            let toggleHtml = `<div style="margin-top: 10px; text-align: right;"><button id="sphere-toggle" style="padding: 4px 8px; border: 1px solid ${toggleColor}; background: white; color: ${toggleColor}; cursor: pointer; font-size: 15px; border-radius: 4px;">${toggleText}</button></div>`;
            
            return subTabsHtml + tableHtml + toggleHtml;
        }
        
        function createBondSubTabs() {
            const bondPropKeys = Object.keys(validBondProperties);
            if (bondPropKeys.length === 0) {
                return '<div style="padding: 20px; text-align: center; color: #666;">No valid bond properties available</div>';
            }
            
            let subTabsHtml = '<div style="display: flex; flex-wrap: wrap; border: 2px solid #4CAF50; border-radius: 8px; margin-bottom: 10px; overflow: hidden;">';
            bondPropKeys.forEach((key, index) => {
                const isFirst = index === 0;
                const isLast = index === bondPropKeys.length - 1;
                const isSelected = currentBondProperty === key;
                
                let borderRadius = '';
                if (isFirst && isLast) borderRadius = '6px';
                else if (isFirst) borderRadius = '6px 0 0 6px';
                else if (isLast) borderRadius = '0 6px 6px 0';
                
                const bgColor = isSelected ? '#e8f5e8' : 'white';
                const textColor = isSelected ? '#2e7d32' : 'black';
                
                subTabsHtml += `<button class="bond-prop-tab" data-property="${key}" style="flex: 1; min-width: 80px; padding: 6px 8px; border: none; background: ${bgColor}; color: ${textColor}; cursor: pointer; font-size: 16.5px; border-radius: ${borderRadius};">${toTitleCase(key)}</button>`;
            });
            subTabsHtml += '</div>';
            
            const selectedProp = currentBondProperty || bondPropKeys[0];
            currentBondProperty = selectedProp;
            
            const values = parsePropertyList(validBondProperties[selectedProp]);
            
            let tableHtml = '<table style="width: 100%; border-collapse: collapse; font-size: 16.5px;">';
            tableHtml += '<thead><tr><th style="border-bottom: 1px solid #ddd; padding: 8px; text-align: left; color: #2e7d32;">Bond</th><th id="bond-value-header" style="border-bottom: 1px solid #ddd; padding: 8px; text-align: right; color: #2e7d32; cursor: pointer; user-select: none;">Value <span id="sort-arrow" style="font-size: 12px;">↕</span></th></tr></thead>';
            tableHtml += '<tbody id="bond-table-body">';
            
            // Create array of valid bond data for sorting
            const bondTableData = [];
            values.forEach((value, index) => {
                // Skip NaN values
                if (value.toLowerCase() === 'nan' || value.toLowerCase() === 'n/a' || value.toLowerCase() === 'none' || value.trim() === '') {
                    return;
                }
                
                const bondName = bondData[index] ? bondData[index].name : `B${index + 1}`;
                bondTableData.push({
                    index: index,
                    name: bondName,
                    value: value,
                    numericValue: parseFloat(value)
                });
            });
            
            // Generate table rows
            bondTableData.forEach(bond => {
                const isSelected = selectedBondIndex === bond.index;
                const bgColor = isSelected ? '#e0e0e0' : 'transparent';
                tableHtml += `<tr class="bond-row" data-bond-index="${bond.index}" data-numeric-value="${bond.numericValue}" style="background-color: ${bgColor}; cursor: pointer;"><td style="border-bottom: 1px solid #eee; padding: 6px; font-weight: bold;">${bond.name}</td><td style="border-bottom: 1px solid #eee; padding: 6px; text-align: right;">${bond.value}</td></tr>`;
            });
            
            tableHtml += '</tbody></table>';
            
            // Create toggle button at bottom right
            const toggleText = showPropertyCylinders ? 'Hide Cylinders' : 'Show Cylinders';
            const toggleColor = showPropertyCylinders ? '#2e7d32' : '#666';
            let toggleHtml = `<div style="margin-top: 10px; text-align: right;"><button id="cylinder-toggle" style="padding: 4px 8px; border: 1px solid ${toggleColor}; background: white; color: ${toggleColor}; cursor: pointer; font-size: 15px; border-radius: 4px;">${toggleText}</button></div>`;
            
            return subTabsHtml + tableHtml + toggleHtml;
        }
        
        function switchTab(tab) {
            currentTab = tab;
            if (tab === 'molecule') {
                molTab.style.background = '#e8f5e8';
                molTab.style.color = '#2e7d32';
                atomTab.style.background = 'white';
                atomTab.style.color = 'black';
                bondTab.style.background = 'white';
                bondTab.style.color = 'black';
                propertiesContent.innerHTML = cachedMolProps || 'No molecule properties available';
                
                // Clear property spheres when switching to molecule tab, but keep selected atom highlighting
                if (viewer) {
                    updateMoleculeView();
                }
            } else if (tab === 'atom') {
                atomTab.style.background = '#e8f5e8';
                atomTab.style.color = '#2e7d32';
                molTab.style.background = 'white';
                molTab.style.color = 'black';
                bondTab.style.background = 'white';
                bondTab.style.color = 'black';
                
                const atomPropKeys = Object.keys(validAtomProperties);
                if (atomPropKeys.length > 0 && !currentAtomProperty) {
                    currentAtomProperty = atomPropKeys[0];
                }
                updateAtomContent();
            } else if (tab === 'bond') {
                bondTab.style.background = '#e8f5e8';
                bondTab.style.color = '#2e7d32';
                molTab.style.background = 'white';
                molTab.style.color = 'black';
                atomTab.style.background = 'white';
                atomTab.style.color = 'black';
                
                const bondPropKeys = Object.keys(validBondProperties);
                if (bondPropKeys.length > 0 && !currentBondProperty) {
                    currentBondProperty = bondPropKeys[0];
                }
                updateBondContent();
            }
        }
        
        function updateAtomContent() {
            const atomContent = createAtomSubTabs();
            propertiesContent.innerHTML = atomContent;
            
            // Add toggle button event listener
            const toggleButton = propertiesContent.querySelector('#sphere-toggle');
            if (toggleButton) {
                toggleButton.addEventListener('click', function() {
                    showPropertySpheres = !showPropertySpheres;
                    updateAtomContent(); // Refresh to update button text and visualization
                });
            }
            
            const atomPropTabs = propertiesContent.querySelectorAll('.atom-prop-tab');
            atomPropTabs.forEach(tab => {
                tab.addEventListener('click', function() {
                    currentAtomProperty = this.dataset.property;
                    atomSortState = 'none'; // Reset sort state when switching properties
                    updateAtomContent();
                });
                addAtomTabHoverEffects(tab);
            });
            
            // Add sort functionality to Value header
            const valueHeader = propertiesContent.querySelector('#atom-value-header');
            const sortArrow = propertiesContent.querySelector('#atom-sort-arrow');
            if (valueHeader && sortArrow) {
                // Update arrow display based on current sort state
                if (atomSortState === 'asc') {
                    sortArrow.innerHTML = '↑';
                } else if (atomSortState === 'desc') {
                    sortArrow.innerHTML = '↓';
                } else {
                    sortArrow.innerHTML = '↕';
                }
                
                valueHeader.addEventListener('click', function() {
                    // Cycle through sort states: none -> asc -> desc -> none
                    if (atomSortState === 'none') {
                        atomSortState = 'asc';
                    } else if (atomSortState === 'asc') {
                        atomSortState = 'desc';
                    } else {
                        atomSortState = 'none';
                    }
                    sortAtomTable();
                });
            }
            
            // Add click handlers to atom rows
            const atomRows = propertiesContent.querySelectorAll('.atom-row');
            atomRows.forEach(row => {
                row.addEventListener('click', function() {
                    const atomIndex = parseInt(this.dataset.atomIndex);
                    selectAtom(atomIndex);
                });
                
                // Add hover effects to rows
                row.addEventListener('mouseenter', function() {
                    if (parseInt(this.dataset.atomIndex) !== selectedAtomIndex) {
                        this.style.backgroundColor = '#f0f0f0';
                    }
                });
                
                row.addEventListener('mouseleave', function() {
                    if (parseInt(this.dataset.atomIndex) !== selectedAtomIndex) {
                        this.style.backgroundColor = 'transparent';
                    }
                });
            });
            
            // Add property visualization when content updates
            addPropertyVisualization();
        }
        
        function updateBondContent() {
            const bondContent = createBondSubTabs();
            propertiesContent.innerHTML = bondContent;
            
            // Add toggle button event listener
            const toggleButton = propertiesContent.querySelector('#cylinder-toggle');
            if (toggleButton) {
                toggleButton.addEventListener('click', function() {
                    showPropertyCylinders = !showPropertyCylinders;
                    updateBondContent(); // Refresh to update button text and visualization
                });
            }
            
            const bondPropTabs = propertiesContent.querySelectorAll('.bond-prop-tab');
            bondPropTabs.forEach(tab => {
                tab.addEventListener('click', function() {
                    currentBondProperty = this.dataset.property;
                    bondSortState = 'none'; // Reset sort state when switching properties
                    updateBondContent();
                });
                addBondTabHoverEffects(tab);
            });
            
            // Add sort functionality to Value header
            const valueHeader = propertiesContent.querySelector('#bond-value-header');
            const sortArrow = propertiesContent.querySelector('#sort-arrow');
            if (valueHeader && sortArrow) {
                // Update arrow display based on current sort state
                if (bondSortState === 'asc') {
                    sortArrow.innerHTML = '↑';
                } else if (bondSortState === 'desc') {
                    sortArrow.innerHTML = '↓';
                } else {
                    sortArrow.innerHTML = '↕';
                }
                
                valueHeader.addEventListener('click', function() {
                    // Cycle through sort states: none -> asc -> desc -> none
                    if (bondSortState === 'none') {
                        bondSortState = 'asc';
                    } else if (bondSortState === 'asc') {
                        bondSortState = 'desc';
                    } else {
                        bondSortState = 'none';
                    }
                    sortBondTable();
                });
            }
            
            // Add click handlers to bond rows
            const bondRows = propertiesContent.querySelectorAll('.bond-row');
            bondRows.forEach(row => {
                row.addEventListener('click', function() {
                    const bondIndex = parseInt(this.dataset.bondIndex);
                    selectBond(bondIndex);
                });
                
                // Add hover effects to rows
                row.addEventListener('mouseenter', function() {
                    if (parseInt(this.dataset.bondIndex) !== selectedBondIndex) {
                        this.style.backgroundColor = '#f0f0f0';
                    }
                });
                
                row.addEventListener('mouseleave', function() {
                    if (parseInt(this.dataset.bondIndex) !== selectedBondIndex) {
                        this.style.backgroundColor = 'transparent';
                    }
                });
            });
            
            // Add bond property visualization when content updates
            addBondPropertyVisualization();
        }
        
        function sortBondTable() {
            const tableBody = propertiesContent.querySelector('#bond-table-body');
            const sortArrow = propertiesContent.querySelector('#sort-arrow');
            
            if (!tableBody || !sortArrow) return;
            
            const rows = Array.from(tableBody.querySelectorAll('.bond-row'));
            
            if (bondSortState === 'asc') {
                rows.sort((a, b) => {
                    const aVal = parseFloat(a.dataset.numericValue);
                    const bVal = parseFloat(b.dataset.numericValue);
                    return aVal - bVal;
                });
                sortArrow.innerHTML = '↑';
            } else if (bondSortState === 'desc') {
                rows.sort((a, b) => {
                    const aVal = parseFloat(a.dataset.numericValue);
                    const bVal = parseFloat(b.dataset.numericValue);
                    return bVal - aVal;
                });
                sortArrow.innerHTML = '↓';
            } else {
                // Original order - sort by bond index
                rows.sort((a, b) => {
                    const aIndex = parseInt(a.dataset.bondIndex);
                    const bIndex = parseInt(b.dataset.bondIndex);
                    return aIndex - bIndex;
                });
                sortArrow.innerHTML = '↕';
            }
            
            // Clear and re-append sorted rows
            tableBody.innerHTML = '';
            rows.forEach(row => {
                tableBody.appendChild(row);
            });
        }
        
        function sortAtomTable() {
            const tableBody = propertiesContent.querySelector('#atom-table-body');
            const sortArrow = propertiesContent.querySelector('#atom-sort-arrow');
            
            if (!tableBody || !sortArrow) return;
            
            const rows = Array.from(tableBody.querySelectorAll('.atom-row'));
            
            if (atomSortState === 'asc') {
                rows.sort((a, b) => {
                    const aVal = parseFloat(a.dataset.numericValue);
                    const bVal = parseFloat(b.dataset.numericValue);
                    return aVal - bVal;
                });
                sortArrow.innerHTML = '↑';
            } else if (atomSortState === 'desc') {
                rows.sort((a, b) => {
                    const aVal = parseFloat(a.dataset.numericValue);
                    const bVal = parseFloat(b.dataset.numericValue);
                    return bVal - aVal;
                });
                sortArrow.innerHTML = '↓';
            } else {
                // Original order - sort by atom index
                rows.sort((a, b) => {
                    const aIndex = parseInt(a.dataset.atomIndex);
                    const bIndex = parseInt(b.dataset.atomIndex);
                    return aIndex - bIndex;
                });
                sortArrow.innerHTML = '↕';
            }
            
            // Clear and re-append sorted rows
            tableBody.innerHTML = '';
            rows.forEach(row => {
                tableBody.appendChild(row);
            });
        }
        
        function selectAtom(atomIndex) {
            // Toggle selection if clicking the same atom
            if (selectedAtomIndex === atomIndex) {
                selectedAtomIndex = null;
            } else {
                selectedAtomIndex = atomIndex;
            }
            
            // Update table row highlighting if on atoms tab
            if (currentTab === 'atom') {
                const atomRows = propertiesContent.querySelectorAll('.atom-row');
                atomRows.forEach((row) => {
                    const rowAtomIndex = parseInt(row.dataset.atomIndex);
                    if (rowAtomIndex === selectedAtomIndex) {
                        row.style.backgroundColor = '#e0e0e0';
                    } else {
                        row.style.backgroundColor = 'transparent';
                    }
                });
            }
            
            // Update 3D visualization based on current tab
            if (currentTab === 'molecule') {
                updateMoleculeView();
            } else {
                addPropertyVisualization();
            }
        }
        
        function selectBond(bondIndex) {
            // Toggle selection if clicking the same bond
            if (selectedBondIndex === bondIndex) {
                selectedBondIndex = null;
            } else {
                selectedBondIndex = bondIndex;
            }
            
            // Update table row highlighting if on bonds tab
            if (currentTab === 'bond') {
                const bondRows = propertiesContent.querySelectorAll('.bond-row');
                bondRows.forEach((row) => {
                    const rowBondIndex = parseInt(row.dataset.bondIndex);
                    if (rowBondIndex === selectedBondIndex) {
                        row.style.backgroundColor = '#e0e0e0';
                    } else {
                        row.style.backgroundColor = 'transparent';
                    }
                });
            }
            
            // Update 3D visualization
            updateBondVisualization();
        }
        
        function updateMoleculeView() {
            if (!viewer) return;
            
            // Clear all existing shapes first
            viewer.removeAllShapes();
            
            // Reset all atom colors to default first
            viewer.setStyle({}, {
                stick: {radius: 0.1, color: '#333333'}, 
                sphere: {radius: 0.3}
            });
            viewer.setStyle({elem: 'H'}, {
                stick: {radius: 0.1, color: '#333333'}, 
                sphere: {radius: 0.15}
            });
            
            // Set selected atom to dark green if any atom is selected
            if (selectedAtomIndex !== null && selectedAtomIndex >= 0) {
                const atoms = viewer.getModel().selectedAtoms({});
                if (atoms && atoms[selectedAtomIndex]) {
                    const selectedAtom = atoms[selectedAtomIndex];
                    const isHydrogen = selectedAtom.elem === 'H';
                    const sphereRadius = isHydrogen ? 0.15 : 0.3;
                    
                    viewer.setStyle({index: selectedAtomIndex}, {
                        stick: {radius: 0.1, color: '#333333'}, 
                        sphere: {radius: sphereRadius, color: '#2e7d32'}
                    });
                }
            }
            
            viewer.render();
        }
        
        function addPropertyVisualization() {
            if (!viewer || currentTab !== 'atom' || !currentAtomProperty || !validAtomProperties[currentAtomProperty]) {
                return;
            }
            
            try {
                // Clear all existing shapes first
                viewer.removeAllShapes();
                
                // Reset all atom colors to default first
                viewer.setStyle({}, {
                    stick: {radius: 0.1, color: '#333333'}, 
                    sphere: {radius: 0.3}
                });
                viewer.setStyle({elem: 'H'}, {
                    stick: {radius: 0.1, color: '#333333'}, 
                    sphere: {radius: 0.15}
                });
                
                // Set selected atom to dark green
                if (selectedAtomIndex !== null && selectedAtomIndex >= 0) {
                    const atoms = viewer.getModel().selectedAtoms({});
                    if (atoms && atoms[selectedAtomIndex]) {
                        const selectedAtom = atoms[selectedAtomIndex];
                        const isHydrogen = selectedAtom.elem === 'H';
                        const sphereRadius = isHydrogen ? 0.15 : 0.3;
                        
                        viewer.setStyle({index: selectedAtomIndex}, {
                            stick: {radius: 0.1, color: '#333333'}, 
                            sphere: {radius: sphereRadius, color: '#2e7d32'}
                        });
                    }
                }
                
                // Only add property spheres if toggle is enabled
                if (showPropertySpheres) {
                    const values = parsePropertyList(validAtomProperties[currentAtomProperty]).map(v => parseFloat(v.trim()));
                    const atoms = viewer.getModel().selectedAtoms({});
                    
                    if (atoms && atoms.length > 0) {
                        // Calculate size scaling
                        const absValues = values.map(v => Math.abs(v));
                        const maxAbsValue = Math.max(...absValues);
                        const minAbsValue = Math.min(...absValues);
                        const range = maxAbsValue - minAbsValue;
                        
                        if (maxAbsValue > 0) {
                            // Add property visualization spheres for each atom
                            values.forEach((value, index) => {
                                if (index >= atoms.length) return;
                                
                                const atom = atoms[index];
                                const absValue = Math.abs(value);
                                
                                // Scale radius: min = 0.36, max = 0.75 (75% of original 1.0)
                                let radius;
                                if (range === 0) {
                                    radius = 0.55; // Middle value between 0.36 and 0.75
                                } else {
                                    radius = 0.36 + ((absValue - minAbsValue) / range) * 0.39;
                                }
                                
                                // Color based on sign
                                const color = value >= 0 ? 'red' : 'blue';
                                
                                viewer.addSphere({
                                    center: {x: atom.x, y: atom.y, z: atom.z},
                                    radius: radius,
                                    color: color,
                                    alpha: 0.5
                                });
                            });
                        }
                    }
                }
                
                viewer.render();
            } catch (error) {
                console.error('Error adding property visualization:', error);
            }
        }
        
        function addBondPropertyVisualization() {
            if (!viewer || currentTab !== 'bond' || !currentBondProperty || !validBondProperties[currentBondProperty]) {
                return;
            }
            
            try {
                // Clear all existing shapes first
                viewer.removeAllShapes();
                
                // Reset all atom colors to default first
                viewer.setStyle({}, {
                    stick: {radius: 0.1, color: '#333333'}, 
                    sphere: {radius: 0.3}
                });
                viewer.setStyle({elem: 'H'}, {
                    stick: {radius: 0.1, color: '#333333'}, 
                    sphere: {radius: 0.15}
                });
                
                // Set selected bond atoms to dark green
                if (selectedBondIndex !== null && selectedBondIndex >= 0 && bondData[selectedBondIndex]) {
                    const bond = bondData[selectedBondIndex];
                    const atoms = viewer.getModel().selectedAtoms({});
                    
                    if (atoms && bond.atom1 < atoms.length && bond.atom2 < atoms.length) {
                        const atom1 = atoms[bond.atom1];
                        const atom2 = atoms[bond.atom2];
                        const isHydrogen1 = atom1.elem === 'H';
                        const isHydrogen2 = atom2.elem === 'H';
                        const sphereRadius1 = isHydrogen1 ? 0.15 : 0.3;
                        const sphereRadius2 = isHydrogen2 ? 0.15 : 0.3;
                        
                        viewer.setStyle({index: bond.atom1}, {
                            stick: {radius: 0.1, color: '#333333'}, 
                            sphere: {radius: sphereRadius1, color: '#2e7d32'}
                        });
                        viewer.setStyle({index: bond.atom2}, {
                            stick: {radius: 0.1, color: '#333333'}, 
                            sphere: {radius: sphereRadius2, color: '#2e7d32'}
                        });
                    }
                }
                
                // Only add property cylinders if toggle is enabled
                if (showPropertyCylinders) {
                    const values = parsePropertyList(validBondProperties[currentBondProperty]);
                    const atoms = viewer.getModel().selectedAtoms({});
                    
                    if (atoms && atoms.length > 0 && bondData.length > 0) {
                        // Filter out bonds with valid numeric values only
                        const validBonds = [];
                        values.forEach((value, index) => {
                            if (index >= bondData.length) return;
                            
                            const numericValue = parseFloat(value);
                            if (!isNaN(numericValue) && isFinite(numericValue) && 
                                value.toLowerCase() !== 'nan' && value.toLowerCase() !== 'n/a' && value.toLowerCase() !== 'none' && value !== '') {
                                validBonds.push({
                                    index: index,
                                    value: numericValue,
                                    bond: bondData[index]
                                });
                            }
                        });
                        
                        if (validBonds.length > 0) {
                            // Calculate opacity scaling based on absolute values of valid bonds only
                            const absValues = validBonds.map(b => Math.abs(b.value));
                            const maxAbsValue = Math.max(...absValues);
                            const minAbsValue = Math.min(...absValues);
                            const range = maxAbsValue - minAbsValue;
                            
                            // Add property visualization cylinders for each valid bond
                            validBonds.forEach((bondInfo) => {
                                const absValue = Math.abs(bondInfo.value);
                                
                                // Calculate opacity: max abs value = 0.8, min abs value = 0.3
                                let alpha;
                                if (range === 0) {
                                    alpha = 0.55; // Middle value between 0.8 and 0.3
                                } else {
                                    // Invert the calculation: higher abs value = higher alpha (more solid)
                                    alpha = 0.3 + ((absValue - minAbsValue) / range) * 0.5;
                                }
                                
                                // Get atom positions for the bond
                                const bond = bondInfo.bond;
                                if (bond.atom1 < atoms.length && bond.atom2 < atoms.length) {
                                    const atom1 = atoms[bond.atom1];
                                    const atom2 = atoms[bond.atom2];
                                    
                                    // Color based on sign (same as atoms: red positive, blue negative)
                                    const color = bondInfo.value >= 0 ? 'red' : 'blue';
                                    
                                    viewer.addCylinder({
                                        start: {x: atom1.x, y: atom1.y, z: atom1.z},
                                        end: {x: atom2.x, y: atom2.y, z: atom2.z},
                                        radius: 0.15,  // Same as hydrogen sphere radius
                                        color: color,
                                        alpha: alpha
                                    });
                                }
                            });
                        }
                    }
                }
                
                viewer.render();
            } catch (error) {
                console.error('Error adding bond property visualization:', error);
            }
        }
        
        function updateBondVisualization() {
            if (!viewer) return;
            
            // Use the bond property visualization function which already handles selection highlighting
            addBondPropertyVisualization();
        }
        
        function addHoverEffects(button) {
            button.addEventListener('mouseenter', function() {
                if (button === molTab && currentTab !== 'molecule') {
                    button.style.background = '#e8f5e8';
                    button.style.color = '#2e7d32';
                } else if (button === atomTab && currentTab !== 'atom') {
                    button.style.background = '#e8f5e8';
                    button.style.color = '#2e7d32';
                } else if (button === bondTab && currentTab !== 'bond') {
                    button.style.background = '#e8f5e8';
                    button.style.color = '#2e7d32';
                }
            });
            
            button.addEventListener('mouseleave', function() {
                if (button === molTab && currentTab !== 'molecule') {
                    button.style.background = 'white';
                    button.style.color = 'black';
                } else if (button === atomTab && currentTab !== 'atom') {
                    button.style.background = 'white';
                    button.style.color = 'black';
                } else if (button === bondTab && currentTab !== 'bond') {
                    button.style.background = 'white';
                    button.style.color = 'black';
                }
            });
        }
        
        function addAtomTabHoverEffects(button) {
            button.addEventListener('mouseenter', function() {
                if (this.dataset.property !== currentAtomProperty) {
                    this.style.background = '#e8f5e8';
                    this.style.color = '#2e7d32';
                }
            });
            
            button.addEventListener('mouseleave', function() {
                if (this.dataset.property !== currentAtomProperty) {
                    this.style.background = 'white';
                    this.style.color = 'black';
                }
            });
        }
        
        function addBondTabHoverEffects(button) {
            button.addEventListener('mouseenter', function() {
                if (this.dataset.property !== currentBondProperty) {
                    this.style.background = '#e8f5e8';
                    this.style.color = '#2e7d32';
                }
            });
            
            button.addEventListener('mouseleave', function() {
                if (this.dataset.property !== currentBondProperty) {
                    this.style.background = 'white';
                    this.style.color = 'black';
                }
            });
        }
        
        function updateMolecule() {
            if (!viewer) return;
            
            const sdf_data = model.get("sdf_data");
            const properties = model.get("properties");
            
            currentAtomProperty = null;
            currentBondProperty = null;
            selectedAtomIndex = null;
            selectedBondIndex = null;
            validAtomProperties = {};
            invalidAtomProperties = {};
            validBondProperties = {};
            invalidBondProperties = {};
            bondData = [];
            
            if (sdf_data) {
                try {
                    atomElements = extractAtomElements(sdf_data);
                    bondData = extractBondData(sdf_data);
                    
                    viewer.clear();
                    viewer.addModel(sdf_data, "sdf");
                    viewer.setStyle({}, {
                        stick: {radius: 0.1, color: '#333333'}, 
                        sphere: {radius: 0.3}
                    });
                    viewer.setStyle({elem: 'H'}, {
                        stick: {radius: 0.1, color: '#333333'}, 
                        sphere: {radius: 0.15}
                    });
                    
                    // Add hover functionality for atoms
                    viewer.setHoverable({}, true, function(atom, viewer, event, container) {
                        if (atom) {
                            const atomIndex = atom.index || atom.serial;
                            const atomName = getAtomNameByIndex(atomIndex);
                            atomLabel.textContent = atomName;
                            atomLabel.style.display = 'block';
                        }
                    }, function(atom, viewer, event, container) {
                        atomLabel.style.display = 'none';
                    });
                    
                    // Add hover functionality for bonds
                    viewer.setHoverable({bond: true}, true, function(bond, viewer, event, container) {
                        if (bond && bond.atom1 && bond.atom2) {
                            const atom1Index = bond.atom1.index !== undefined ? bond.atom1.index : bond.atom1.serial;
                            const atom2Index = bond.atom2.index !== undefined ? bond.atom2.index : bond.atom2.serial;
                            const atom1Name = getAtomNameByIndex(atom1Index);
                            const atom2Name = getAtomNameByIndex(atom2Index);
                            atomLabel.textContent = `${atom1Name}-${atom2Name}`;
                            atomLabel.style.display = 'block';
                        }
                    }, function(bond, viewer, event, container) {
                        atomLabel.style.display = 'none';
                    });
                    
                    // Add click functionality to atoms - NOW WORKS ON BOTH TABS
                    viewer.setClickable({}, true, function(atom, viewer, event, container) {
                        if (atom) {
                            const atomIndex = atom.index || atom.serial;
                            selectAtom(atomIndex);
                        }
                    });
                    
                    // Fit and zoom molecule to optimal viewing level
                    viewer.zoomTo();
                    viewer.zoom(2); // Zoom in for better visibility
                    viewer.render();
                } catch (error) {
                    console.error('Error rendering molecule:', error);
                }
            }
            
            if (properties && Object.keys(properties).length > 0) {
                const molNumericalProps = {};
                const molStringProps = {};
                const numAtoms = model.get("num_atoms") || 0;
                const numAtomsNoH = model.get("num_atoms_no_h") || 0;
                const numBonds = bondData.length;
                
                for (const [key, value] of Object.entries(properties)) {
                    if (!key.startsWith('mol_') && !key.startsWith('atom_') && !key.startsWith('bond_')) {
                        continue;
                    }
                    
                    const displayKey = key.replace(/^(mol_|atom_|bond_)/, '');
                    
                    if (key.startsWith('atom_') && typeof value === 'string' && value.includes(',')) {
                        const parts = parsePropertyList(value);
                        if (parts.every(v => v.toLowerCase() === 'nan' || v.toLowerCase() === 'n/a' || v.toLowerCase() === 'none' || v === '' || (!isNaN(parseFloat(v)) && isFinite(parseFloat(v))))) {
                            if (parts.length === numAtoms || parts.length === numAtomsNoH) {
                                validAtomProperties[displayKey] = value;
                            }
                            continue;
                        }
                    }
                    
                    if (key.startsWith('bond_') && typeof value === 'string' && value.includes(',')) {
                        const parts = parsePropertyList(value);
                        if (parts.every(v => v.toLowerCase() === 'nan' || v.toLowerCase() === 'n/a' || v.toLowerCase() === 'none' || v === '' || (!isNaN(parseFloat(v)) && isFinite(parseFloat(v))))) {
                            if (parts.length === numBonds) {
                                validBondProperties[displayKey] = value;
                            }
                            continue;
                        }
                    }
                    
                    if (key.startsWith('mol_')) {
                        const numValue = parseFloat(value);
                        if (!isNaN(numValue) && isFinite(numValue) && value.toString().trim() !== '') {
                            molNumericalProps[displayKey] = value;
                        } else {
                            molStringProps[displayKey] = value;
                        }
                    }
                }
                
                let molPropsHtml = '';
                if (Object.keys(molNumericalProps).length > 0) {
                    for (const [key, value] of Object.entries(molNumericalProps)) {
                        const titleCaseKey = toTitleCase(key);
                        molPropsHtml += `<div style="margin-bottom: 8px;"><strong style="color: #2e7d32;">${titleCaseKey}</strong> ${value}</div>`;
                    }
                }
                
                if (Object.keys(molNumericalProps).length > 0 && Object.keys(molStringProps).length > 0) {
                    molPropsHtml += '<hr style="border: none; border-top: 1px solid #2e7d32; margin: 15px 0;">';
                }
                
                if (Object.keys(molStringProps).length > 0) {
                    for (const [key, value] of Object.entries(molStringProps)) {
                        const titleCaseKey = toTitleCase(key);
                        molPropsHtml += `<div style="margin-bottom: 8px;"><strong style="color: #2e7d32;">${titleCaseKey}</strong> ${value}</div>`;
                    }
                }
                
                cachedMolProps = molPropsHtml || 'No molecule properties available';
                switchTab(currentTab);
            } else {
                cachedMolProps = 'No properties available';
                propertiesContent.innerHTML = 'No properties available';
            }
        }
        
        // Initialize everything
        ensure3Dmol().then(() => {
            try {
                viewer = window.$3Dmol.createViewer(el.querySelector('#' + viewerId), {
                    defaultcolors: window.$3Dmol.elementColors.defaultColors
                });
                
                viewer.setHoverDuration(100);
                
                addHoverEffects(molTab);
                addHoverEffects(atomTab);
                addHoverEffects(bondTab);
                
                molTab.addEventListener('click', () => switchTab('molecule'));
                atomTab.addEventListener('click', () => switchTab('atom'));
                bondTab.addEventListener('click', () => switchTab('bond'));
                
                model.on("change:sdf_data", updateMolecule);
                model.on("change:properties", updateMolecule);
                
                updateMolecule();
                
            } catch (error) {
                propertiesContent.innerHTML = 'Error initializing viewer: ' + error.message;
            }
        }).catch(error => {
            propertiesContent.innerHTML = 'Error loading 3Dmol.js: ' + error.message;
        });
    }
    
    export default { render };
    """

    _css = """
    .widget-output {
        border: 1px solid #ccc;
        border-radius: 4px;
        padding: 10px;
    }
    """

    # Traits to hold the data
    sdf_data = traitlets.Unicode("").tag(sync=True)
    properties = traitlets.Dict({}).tag(sync=True)
    num_atoms = traitlets.Int(0).tag(sync=True)
    num_atoms_no_h = traitlets.Int(0).tag(sync=True)

    def __init__(self, molecule=None, **kwargs):
        super().__init__(**kwargs)
        if molecule is not None:
            self.set_molecule(molecule)

    def set_molecule(self, mol):
        """Set the molecule to display"""
        if mol is None:
            self.sdf_data = ""
            self.properties = {}
            self.num_atoms = 0
            self.num_atoms_no_h = 0
            return

        # Get atom counts before any modifications
        mol_no_h = Chem.RemoveHs(mol)
        self.num_atoms_no_h = mol_no_h.GetNumAtoms()

        # Generate 3D coordinates if not present
        if not mol.GetNumConformers():
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol)
            AllChem.MMFFOptimizeMolecule(mol)

        # Get total atom count after adding hydrogens
        self.num_atoms = mol.GetNumAtoms()

        # Convert to SDF format
        self.sdf_data = Chem.MolToMolBlock(mol)

        # Get molecule properties
        self.properties = mol.GetPropsAsDict()


# Example usage:
# from rdkit import Chem
# mol = Chem.MolFromSmiles('CCO')  # ethanol
# widget = MoleculeViewer(mol)
# widget
